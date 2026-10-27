"""
Esperimento (punto 3): la pioggia (rr7) si corregge meglio cambiando il modo in cui la rete la impara?
Stesse feature del modello attuale, CV regione x anno di cv_comune.py, 2 seed per variante.

  P0 residuo           come oggi: la rete impara obs - aurora con errore quadratico
  P1 radice            la rete impara sqrt(obs) - sqrt(aurora): pesa meno i pochi eventi estremi
  P2 log               la rete impara log(1+obs) - log(1+aurora): come P1, ancora piu' marcato
  P3 due parti         una rete stima la probabilita' di giorno piovoso (>= 1 mm), una seconda la
                       quantita' nei giorni piovosi; previsione = probabilita' x quantita'

Con l'errore quadratico la P0 stima gia' la media condizionata, che e' cio' che minimizza l'RMSE:
P1-P3 possono migliorare altri aspetti (errore assoluto, riconoscere i giorni di pioggia) anche se
l'RMSE non scende. Per questo oltre a RMSE, bias e MAE si misurano, per le soglie 1 e 20 mm:
  POD  frazione dei giorni con pioggia >= soglia riconosciuti
  FAR  frazione dei giorni previsti >= soglia in cui non e' piovuto cosi'
  CSI  indice complessivo (colpi / (colpi + mancati + falsi allarmi))

Uso:
    python esperimento_pioggia.py
Output:
    esperimento_pioggia_risultati.csv
"""
import numpy as np
import pandas as pd
import torch

from cv_comune import BATCH, Rete, allena_valuta, carica_tabella, errori, folds
from train_downscaling_head import FEATURE_COLS

SEEDS = [0, 1]
MAX_EPOCHE = 25
SOGLIE = [1.0, 20.0]
SOGLIA_PIOVOSO = 1.0


def categoriche(pred, obs):
    out = {}
    for s in SOGLIE:
        sp, so = pred >= s, obs >= s
        colpi, mancati, falsi = (sp & so).sum(), (~sp & so).sum(), (sp & ~so).sum()
        out[f"pod_{s:g}"] = colpi / max(colpi + mancati, 1)
        out[f"far_{s:g}"] = falsi / max(colpi + falsi, 1)
        out[f"csi_{s:g}"] = colpi / max(colpi + mancati + falsi, 1)
    return out


def due_parti(tr, va, seed):
    """Classificatore (piovoso si/no) + regressione della quantita' sui soli giorni piovosi."""
    torch.manual_seed(seed)
    m = tr[FEATURE_COLS].values.mean(axis=0)
    s = tr[FEATURE_COLS].values.std(axis=0)
    s[s == 0] = 1
    X = lambda d: torch.tensor((d[FEATURE_COLS].values - m) / s, dtype=torch.float32)
    x, xv = X(tr), X(va)
    piov = torch.tensor((tr["obs"].values >= SOGLIA_PIOVOSO).astype("float32")).unsqueeze(1)
    res = torch.tensor(tr["residual"].values, dtype=torch.float32).unsqueeze(1)
    bagnati = torch.where(piov.squeeze(1) > 0)[0]
    cls, reg = Rete(len(FEATURE_COLS)), Rete(len(FEATURE_COLS))
    oc = torch.optim.Adam(cls.parameters(), lr=1e-3, weight_decay=1e-4)
    orr = torch.optim.Adam(reg.parameters(), lr=1e-3, weight_decay=1e-4)
    av = va["aurora"].values
    preds = []
    for _ in range(MAX_EPOCHE):
        cls.train(); reg.train()
        perm = torch.randperm(len(x))
        for i in range(0, len(x), BATCH):
            b = perm[i:i + BATCH]
            oc.zero_grad()
            torch.nn.functional.binary_cross_entropy_with_logits(cls(x[b]), piov[b]).backward()
            oc.step()
        perm = bagnati[torch.randperm(len(bagnati))]
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            orr.zero_grad()
            torch.nn.functional.mse_loss(reg(x[b]), res[b]).backward()
            orr.step()
        cls.eval(); reg.eval()
        with torch.no_grad():
            p = torch.sigmoid(cls(xv)).numpy().ravel()
            q = np.clip(av + reg(xv).numpy().ravel(), 0, None)
        preds.append((p * q).astype("float32"))
    return np.array(preds)


def main():
    torch.set_num_threads(6)
    t = carica_tabella()
    tp = t[t["parameter"] == "rr7"]
    righe = []
    for reg, anno, tr, va in folds(tp):
        ov = va["obs"].values
        for seed in SEEDS:
            varianti = {}
            for nome, trasf in [("P0 residuo", "nessuna"), ("P1 radice", "radice"), ("P2 log", "log1p")]:
                _, varianti[nome] = allena_valuta(tr, va, FEATURE_COLS, "rr7", max_epoche=MAX_EPOCHE, seed=seed,
                                                  trasformazione=trasf, salva_pred=True)
            varianti["P3 due parti"] = due_parti(tr, va, seed)
            for nome, pred in varianti.items():
                for ep, p in enumerate(pred, start=1):
                    righe.append({"variante": nome, "regione": reg, "anno": anno, "seed": seed, "epoca": ep,
                                  **errori(p, ov), **categoriche(p, ov)})
        print(f"{reg} {anno} fatto", flush=True)
        pd.DataFrame(righe).to_csv("esperimento_pioggia_risultati.csv", index=False)

    res = pd.DataFrame(righe)
    # epoca scelta per variante sull'RMSE medio (come per il modello), poi tutte le metriche a quell'epoca
    curva = res.groupby(["variante", "epoca"]).mean(numeric_only=True)
    ep = curva["rmse"].groupby("variante").idxmin()
    tab = pd.DataFrame({v: curva.loc[e] for v, e in ep.items()}).T.drop(columns=["anno", "seed"])
    tab.insert(0, "epoca", [e[1] for e in ep])
    rif = []
    for reg, anno, _, va in folds(tp):
        rif.append({**errori(va["aurora"].values, va["obs"].values), **categoriche(va["aurora"].values, va["obs"].values)})
    tab.loc["Aurora interpolata"] = pd.DataFrame(rif).mean()
    pd.set_option("display.width", 200)
    print("\n=== rr7, CV regione x anno: medie sui fold e sui seed, all'epoca con RMSE migliore ===")
    print(tab.round(3).to_string())


if __name__ == "__main__":
    main()
