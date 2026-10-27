"""
Testa di downscaling (bias-correction) sulla previsione Aurora, nello stile "Granite": una piccola
rete impara il residuo osservazione - Aurora (interpolata bilinearmente nel punto), usando come
feature Aurora stessa, le altre variabili di Aurora nel punto, la quota (DEM 30 m), la stagione e,
per tn3 e qq2, la forma del rilievo. All'inferenza valore_corretto = aurora + correzione: la rete non
reinventa il segnale meteorologico, solo la correzione locale che Aurora (0.25 gradi) non vede.

Modello finale (scelto con gli esperimenti in cv_comune.py e esperimento_*.py):
  - una rete per variabile, 64-64, ReLU, dropout 0.1 (esperimento_iperparametri.py)
  - ensemble: N_SEED reti identiche allenate con seed diversi, previsione = media (esperimento_ensemble.py)
  - feature: FEATURE_COLS (28) senza a_<variabile> (copia di "aurora") + FORMA_RILIEVO per tn3 e qq2
    (esperimento_terreno.py)
  - pioggia: residuo semplice con errore quadratico (esperimento_pioggia.py)

Input atteso nella working directory:
  - training_table_aurora.parquet    (dal notebook padania_cerra_ecad_analysis §6.6)
  - test_table_aurora.parquet        (stesso notebook: Piemonte 2020, SOLO per la valutazione finale)
  - esclusioni_training.csv          (notebook §7.2, opzionale: righe da non usare)
  - epoche_cv.json                   (cv_regione_anno.py)

Uso:
  python cv_regione_anno.py                          # una volta: sceglie le epoche (epoche_cv.json)
  python train_downscaling_head.py                   # modello finale, tutte e 4 le variabili
  python train_downscaling_head.py --valuta_test     # UNA volta sola, a modello definitivo

Output:
  - checkpoints/{parameter}_finale.pt: le N_SEED reti, le feature e la loro normalizzazione
    (medie/deviazioni del training), necessarie all'inferenza (mappa_downscaling.py)
  - stampa RMSE/bias di Aurora interpolata e del downscaling (train; test solo con --valuta_test)
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    import pytorch_lightning as pl
except ImportError as e:
    raise SystemExit(
        "manca pytorch_lightning. Su Lightning AI Studio: pip install pytorch-lightning"
    ) from e


# --------------------------------------------------------------------------- #
# Dati
# --------------------------------------------------------------------------- #

# lat/lon tolte: l'ablazione leave-one-region-out le mostra come identificatori di
# regione che non si trasferiscono tra regioni (migliorano o pareggiano in 11
# casi su 12, vedi ablation_latlon_all.csv). Restano il valore Aurora, la quota
# DTM e la stagione, cioe' le informazioni fisiche.
# Le altre variabili di Aurora nel punto della stazione (interpolate bilinearmente dai campi,
# notebook §6.6): l'elenco lo scrive il notebook in fattori_scala_aurora.json, cosi' e' definito in un
# solo punto. Con queste variabili l'errore scende del 12-16% su tutti i parametri, in tutti i fold
# della cross-validation regione x anno (esperimento_bilineare.py).
AURORA_INFO = "fattori_scala_aurora.json"
if os.path.exists(AURORA_INFO):
    AURORA_FEATURE = json.load(open(AURORA_INFO))["aurora_feature"]
else:
    AURORA_FEATURE = []
    print(f"ATTENZIONE: {AURORA_INFO} non trovato (notebook §6.6): solo aurora, quota e stagione come feature.")
FEATURE_COLS = ["aurora", "elev", "doy_sin", "doy_cos"] + AURORA_FEATURE
# "rilievo_locale" (max - min di quota entro 500 m, inutili/add_terrain_feature.py) e' stato testato
# e scartato con il modello precedente.
# Forma del rilievo (descrittori_terreno.py): solo per tn3 (aria fredda in conche e fondovalle) e
# qq2 (ombreggiamento); per tx3 e rr7 nessun guadagno nella cross-validation (esperimento_terreno.py).
FORMA_RILIEVO = ["tpi_1km", "tpi_5km", "tpi_20km", "pendenza", "nordness", "eastness"]  # = descrittori_terreno
FEATURE_EXTRA = {"tn3": FORMA_RILIEVO, "qq2": FORMA_RILIEVO}
PARAMS = ["tx3", "tn3", "rr7", "qq2"]

# Rete e allenamento del modello finale
HIDDEN = (64, 64)
DROPOUT = 0.1
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH = 4096
N_SEED = 5


def feature_cols(param):
    """Feature del modello per la variabile param. a_<param> e' tolta: per la variabile stessa e' una
    copia esatta di "aurora" (in FEATURE_COLS resta per gli esperimenti precedenti, che la includevano)."""
    return [c for c in FEATURE_COLS if c != f"a_{param}"] + FEATURE_EXTRA.get(param, [])


def _read_table(basename):
    """Legge basename.parquet (formato del notebook §6.6), oppure .csv.gz / .csv."""
    if os.path.exists(basename + ".parquet"):
        return pd.read_parquet(basename + ".parquet")
    for ext in (".csv.gz", ".csv"):
        p = basename + ext
        if os.path.exists(p):
            return pd.read_csv(p, parse_dates=["date"])
    raise FileNotFoundError(f"non trovo {basename}.parquet / .csv.gz / .csv nella working directory")


def _read_table_optional(basename):
    """Come _read_table, ma restituisce None invece di sollevare un errore
    se il file non c'e' - usato per il test (es. Piemonte-Aurora), che puo'
    non essere ancora pronto senza per questo bloccare train+val."""
    for ext in (".parquet", ".csv.gz", ".csv"):
        if os.path.exists(basename + ext):
            return _read_table(basename)
    print(f"ATTENZIONE: {basename} non trovato - proseguo senza test, solo train+val.")
    return None


ESCLUSIONI = "esclusioni_training.csv"  # generato dal notebook, §7.2


def applica_esclusioni(df, path=ESCLUSIONI):
    """Toglie le righe elencate in esclusioni_training.csv (notebook §7.2): stazioni intere
    (date vuota; parameter vuoto = tutti i parametri) o singoli giorni. La tabella di training
    resta intatta; se il file non c'e' non toglie niente."""
    if not os.path.exists(path):
        print(f"ATTENZIONE: {path} non trovato, nessuna esclusione applicata.")
        return df
    e = pd.read_csv(path, parse_dates=["date"])
    intere = e[e["date"].isna()]
    giorni = e[e["date"].notna()]
    tutte = intere[intere["parameter"].isna()]["uid"]
    per_param = pd.MultiIndex.from_frame(intere[intere["parameter"].notna()][["uid", "parameter"]])
    per_giorno = pd.MultiIndex.from_frame(giorni[["uid", "parameter", "date"]])
    m = (df["uid"].isin(tutte).values
         | pd.MultiIndex.from_frame(df[["uid", "parameter"]]).isin(per_param)
         | pd.MultiIndex.from_frame(df[["uid", "parameter", "date"]]).isin(per_giorno))
    print(f"esclusioni ({path}): tolte {int(m.sum())} righe su {len(df)}")
    return df[~m]


def load_tables(parameter, table="training_table_aurora", test_table="test_table_aurora", usa_esclusioni=True):
    """Il test (Piemonte, split "test") e' in una tabella a parte (notebook §6.6); se manca si
    prosegue con il solo training."""
    train_val = _read_table(table)
    test = _read_table_optional(test_table) if test_table else None

    df = pd.concat([train_val, test], ignore_index=True) if test is not None else train_val.copy()
    df = df[df["parameter"] == parameter].copy()
    if usa_esclusioni:
        df = applica_esclusioni(df)
    if df.empty:
        raise ValueError(f"nessuna riga per parameter={parameter!r}: controlla il nome (tx3/tn3/rr7/qq2)")

    doy = df["date"].dt.dayofyear.astype(float)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["residual"] = df["obs"] - df["aurora"]
    return df


class ResidualDataset(Dataset):
    """Feature -> residuo (obs - aurora). Le feature sono già standardizzate
    (media/std calcolate SOLO sul train, per non far trapelare val/test)."""

    def __init__(self, df, mean, std):
        x = (df[FEATURE_COLS].values - mean) / std
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(df["residual"].values, dtype=torch.float32).unsqueeze(1)
        self.aurora = torch.tensor(df["aurora"].values, dtype=torch.float32).unsqueeze(1)
        self.obs = torch.tensor(df["obs"].values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.aurora[idx], self.obs[idx]


# --------------------------------------------------------------------------- #
# Modello
# --------------------------------------------------------------------------- #

class Rete(nn.Module):
    """Rete del modello finale (e degli esperimenti, via cv_comune.py): piccola apposta, e' una
    correzione locale, non deve reimparare la meteorologia da zero."""

    def __init__(self, n_in, hidden=HIDDEN, dropout=0.0):
        super().__init__()
        strati, n = [], n_in
        for h in hidden:
            strati += [nn.Linear(n, h), nn.ReLU()] + ([nn.Dropout(dropout)] if dropout else [])
            n = h
        self.net = nn.Sequential(*strati, nn.Linear(n, 1))

    def forward(self, x):
        return self.net(x)


class DownscalingHead(pl.LightningModule):
    """Versione precedente della testa (Lightning, 64-64 senza dropout, una rete sola): usata solo
    da esperimento_sarah_c.py. Il modello finale usa Rete."""

    def __init__(self, n_features, hidden=64, lr=1e-3, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)

    def _step(self, batch, stage):
        x, y, aurora, obs = batch
        pred_residual = self(x)
        loss = nn.functional.mse_loss(pred_residual, y)
        with torch.no_grad():
            corrected = aurora + pred_residual
            rmse_head = torch.sqrt(torch.mean((corrected - obs) ** 2))
            rmse_baseline = torch.sqrt(torch.mean((aurora - obs) ** 2))
        self.log(f"{stage}_loss", loss, prog_bar=(stage == "train"), on_epoch=True, on_step=False)
        self.log(f"{stage}_rmse_head", rmse_head, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}_rmse_baseline", rmse_baseline, prog_bar=False, on_epoch=True, on_step=False)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

NON_NEGATIVI = {"rr7", "qq2"}  # pioggia e radiazione: il valore corretto non puo' essere < 0


def allena_rete(tr, feature, epoche, seed, mean, std, hidden=HIDDEN, dropout=DROPOUT, lr=LR, wd=WEIGHT_DECAY,
                batch=BATCH):
    """Allena una rete sul residuo obs - aurora. Stesso ciclo di allenamento della cross-validation
    (cv_comune.allena_valuta): quello che e' stato validato e' esattamente quello che si allena."""
    torch.manual_seed(seed)
    x = torch.tensor((tr[feature].values - mean) / std, dtype=torch.float32)
    y = torch.tensor(tr["obs"].values - tr["aurora"].values, dtype=torch.float32).unsqueeze(1)
    net = Rete(len(feature), hidden, dropout)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epoche):
        net.train()
        perm = torch.randperm(len(y))
        for i in range(0, len(y), batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            nn.functional.mse_loss(net(x[b]), y[b]).backward()
            opt.step()
    return net.eval()


def prevedi(modello, df, param):
    """Valore corretto con l'ensemble: media delle reti (ognuna tagliata a zero per pioggia e
    radiazione, come nella cross-validation). modello = dizionario salvato da main()."""
    x = torch.tensor((df[modello["feature_cols"]].values - np.array(modello["mean"])) / np.array(modello["std"]),
                     dtype=torch.float32)
    prev = []
    for stato in modello["reti"]:
        net = Rete(len(modello["feature_cols"]), tuple(modello["hidden"]), modello["dropout"])
        net.load_state_dict(stato)
        with torch.no_grad():
            p = df["aurora"].values + net.eval()(x).numpy().ravel()
        prev.append(np.clip(p, 0, None) if param in NON_NEGATIVI else p)
    return np.mean(prev, axis=0)


def carica_modello(param):
    return torch.load(f"checkpoints/{param}_finale.pt", weights_only=False)


def evaluate(modello, df, param, label):
    pred, obs, aur = prevedi(modello, df, param), df["obs"].values, df["aurora"].values
    print(f"[{label}] n={len(df)}")
    print(f"  Aurora interpolata -> RMSE={np.sqrt(np.mean((aur - obs) ** 2)):.3f}  bias={np.mean(aur - obs):.3f}")
    print(f"  downscaling        -> RMSE={np.sqrt(np.mean((pred - obs) ** 2)):.3f}  bias={np.mean(pred - obs):.3f}")


def main():
    """Modello finale: per ogni variabile allena N_SEED reti su TUTTE le righe di training (tutti gli
    anni, tutte le regioni, Piemonte escluso), per un numero di epoche fisso scelto dalla
    cross-validation regione x anno (cv_regione_anno.py -> epoche_cv.json), senza early stopping.
    Salva le reti e la normalizzazione delle feature; valuta il test (Piemonte) solo con --valuta_test."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--parameter", choices=PARAMS, default=None, help="una sola variabile (default: tutte)")
    ap.add_argument("--table", default="training_table_aurora",
                    help="basename (senza estensione) della tabella di training")
    ap.add_argument("--test_table", default="test_table_aurora",
                    help="basename (senza estensione) della tabella di test (Piemonte)")
    ap.add_argument("--valuta_test", action="store_true",
                    help="valuta anche sul test Piemonte: da usare UNA volta sola, a modello definitivo")
    ap.add_argument("--senza_esclusioni", action="store_true",
                    help="non applicare esclusioni_training.csv (per confronto)")
    args = ap.parse_args()
    torch.set_num_threads(6)
    if not os.path.exists("epoche_cv.json"):
        raise SystemExit("non trovo epoche_cv.json: lancia prima cv_regione_anno.py")
    epoche = json.load(open("epoche_cv.json"))
    os.makedirs("checkpoints", exist_ok=True)

    for param in ([args.parameter] if args.parameter else PARAMS):
        feature = feature_cols(param)
        df = load_tables(param, table=args.table, test_table=args.test_table,
                         usa_esclusioni=not args.senza_esclusioni)
        mancanti = [c for c in feature if c not in df.columns]
        if mancanti:
            raise SystemExit(f"nella tabella mancano {mancanti}: rilancia il notebook §6.6")
        train_df = df[df["split"] != "test"].dropna(subset=feature + ["obs", "aurora"])
        test_df = df[df["split"] == "test"].dropna(subset=feature + ["obs", "aurora"])
        print(f"\n=== {param}: train={len(train_df)}  test(Piemonte)={len(test_df)}  epoche={epoche[param]}  "
              f"feature={len(feature)}  reti={N_SEED} ===")
        mean = train_df[feature].values.mean(axis=0)
        std = train_df[feature].values.std(axis=0)
        std[std == 0] = 1.0
        reti = [allena_rete(train_df, feature, epoche[param], seed, mean, std).state_dict() for seed in range(N_SEED)]
        modello = {"feature_cols": feature, "mean": mean.tolist(), "std": std.tolist(), "epoche": epoche[param],
                   "hidden": list(HIDDEN), "dropout": DROPOUT, "reti": reti}
        torch.save(modello, f"checkpoints/{param}_finale.pt")
        print(f"salvato checkpoints/{param}_finale.pt")
        evaluate(modello, train_df, param, "train (dati visti in training)")
        if not args.valuta_test:
            print("[test] non valutato: aggiungi --valuta_test una sola volta, quando il modello e' definitivo.")
        elif len(test_df):
            evaluate(modello, test_df, param, "test (Piemonte 2020, regione e anno mai visti)")
        else:
            print("[test] nessuna riga di test disponibile - saltato.")


if __name__ == "__main__":
    main()
