"""
Downscaling (bias-correction) head on the Aurora forecast, in the "Granite" style: a small network
learns the residual observation - Aurora (bilinearly interpolated to the point), using as features
Aurora itself, the other Aurora variables at the point, the elevation (30 m DEM), the season and,
for tn3 and qq2, the terrain shape. At inference corrected_value = aurora + correction: the network
does not reinvent the meteorological signal, only the local correction that Aurora (0.25 degrees)
cannot see.

Final model (chosen with the experiments in cv_comune.py and esperimento_*.py):
  - one network per variable, 64-64, ReLU, dropout 0.1 (esperimento_iperparametri.py)
  - ensemble: N_SEED identical networks trained with different seeds, forecast = mean (esperimento_ensemble.py)
  - features: FEATURE_COLS (28) without a_<variable> (copy of "aurora") + FORMA_RILIEVO for tn3 and qq2
    (esperimento_terreno.py)
  - precipitation: plain residual with squared error (esperimento_pioggia.py)

Expected input in the working directory:
  - training_table_aurora.parquet    (from the data preparation notebook, §6.6)
  - test_table_aurora.parquet        (same notebook: Piedmont 2020, ONLY for the final evaluation)
  - esclusioni_training.csv          (notebook §7.2, optional: rows not to be used)
  - epoche_cv.json                   (cv_regione_anno.py)

Usage:
  python cv_regione_anno.py                          # once: chooses the epochs (epoche_cv.json)
  python train_downscaling_head.py                   # final model, all 4 variables
  python train_downscaling_head.py --valuta_test     # ONCE only, when the model is final

Output:
  - checkpoints/{parameter}_finale.pt: the N_SEED networks, the features and their normalisation
    (training means/standard deviations), needed for inference (mappa_downscaling.py)
  - prints RMSE/bias of interpolated Aurora and of the downscaling (train; test only with --valuta_test)
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

# Paths relative to the main project folder (where the script is run from)
TABELLA_TRAINING = "tabelle/training_table_aurora.parquet"  # notebook §6.6
TABELLA_TEST = "tabelle/test_table_aurora.parquet"
FATTORI_SCALA = "tabelle/fattori_scala_aurora.json"
ESCLUSIONI = "tabelle/esclusioni_training.csv"  # notebook §7.2
EPOCHE_CV = "modello/epoche_cv.json"  # cv_regione_anno.py
CHECKPOINTS = "modello/checkpoints"


def checkpoint(param):
    return os.path.join(CHECKPOINTS, f"{param}_finale.pt")

try:
    import pytorch_lightning as pl
except ImportError as e:
    raise SystemExit(
        "pytorch_lightning is missing. On Lightning AI Studio: pip install pytorch-lightning"
    ) from e


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

# lat/lon removed: the leave-one-region-out ablation shows them acting as region identifiers
# that do not transfer between regions (removing them improves or ties in 11 cases out of 12,
# see ablation_latlon_all.csv). What remains is the Aurora value, the DTM elevation and the
# season, i.e. the physical information.
# The other Aurora variables at the station location (bilinearly interpolated from the fields,
# notebook §6.6): the list is written by the notebook in fattori_scala_aurora.json, so it is defined
# in one place only. With these variables the error drops by 12-16% for all variables, in all the
# folds of the region x year cross-validation (esperimento_bilineare.py).
AURORA_INFO = FATTORI_SCALA
if os.path.exists(AURORA_INFO):
    AURORA_FEATURE = json.load(open(AURORA_INFO))["aurora_feature"]
else:
    AURORA_FEATURE = []
    print(f"WARNING: {AURORA_INFO} not found (notebook §6.6): only aurora, elevation and season as features.")
FEATURE_COLS = ["aurora", "elev", "doy_sin", "doy_cos"] + AURORA_FEATURE
# "rilievo_locale" (max - min elevation within 500 m, inutili/add_terrain_feature.py) was tested
# and discarded with the previous model.
# Terrain shape (descrittori_terreno.py): only for tn3 (cold air in basins and valley floors) and
# qq2 (shading); no gain in cross-validation for tx3 and rr7 (esperimento_terreno.py).
FORMA_RILIEVO = ["tpi_1km", "tpi_5km", "tpi_20km", "pendenza", "nordness", "eastness"]  # = descrittori_terreno
FEATURE_EXTRA = {"tn3": FORMA_RILIEVO, "qq2": FORMA_RILIEVO}
PARAMS = ["tx3", "tn3", "rr7", "qq2"]

# Network and training of the final model
HIDDEN = (64, 64)
DROPOUT = 0.1
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH = 4096
N_SEED = 5


def feature_cols(param):
    """Model features for the variable param. a_<param> is removed: for the variable itself it is an
    exact copy of "aurora" (it stays in FEATURE_COLS for the earlier experiments, which included it)."""
    return [c for c in FEATURE_COLS if c != f"a_{param}"] + FEATURE_EXTRA.get(param, [])


def _read_table(basename):
    """Reads basename.parquet (format of the notebook §6.6), or .csv.gz / .csv."""
    if os.path.exists(basename + ".parquet"):
        return pd.read_parquet(basename + ".parquet")
    for ext in (".csv.gz", ".csv"):
        p = basename + ext
        if os.path.exists(p):
            return pd.read_csv(p, parse_dates=["date"])
    raise FileNotFoundError(f"cannot find {basename}.parquet / .csv.gz / .csv in the working directory")


def _read_table_optional(basename):
    """Like _read_table, but returns None instead of raising an error if the file
    is missing - used for the test table (e.g. Piedmont-Aurora), which may not be
    ready yet without blocking train+val."""
    for ext in (".parquet", ".csv.gz", ".csv"):
        if os.path.exists(basename + ext):
            return _read_table(basename)
    print(f"WARNING: {basename} not found - continuing without test, train+val only.")
    return None




def applica_esclusioni(df, path=ESCLUSIONI):
    """Removes the rows listed in esclusioni_training.csv (notebook §7.2): whole stations
    (empty date; empty parameter = all variables) or single days. The training table itself
    stays intact; if the file is missing nothing is removed."""
    if not os.path.exists(path):
        print(f"WARNING: {path} not found, no exclusion applied.")
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
    print(f"exclusions ({path}): removed {int(m.sum())} rows out of {len(df)}")
    return df[~m]


def load_tables(parameter, table=os.path.splitext(TABELLA_TRAINING)[0], test_table=os.path.splitext(TABELLA_TEST)[0],
                usa_esclusioni=True):
    """The test set (Piedmont, split "test") is in a separate table (notebook §6.6); if it is
    missing, only the training table is used."""
    train_val = _read_table(table)
    test = _read_table_optional(test_table) if test_table else None

    df = pd.concat([train_val, test], ignore_index=True) if test is not None else train_val.copy()
    df = df[df["parameter"] == parameter].copy()
    if usa_esclusioni:
        df = applica_esclusioni(df)
    if df.empty:
        raise ValueError(f"no rows for parameter={parameter!r}: check the name (tx3/tn3/rr7/qq2)")

    doy = df["date"].dt.dayofyear.astype(float)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["residual"] = df["obs"] - df["aurora"]
    return df


class ResidualDataset(Dataset):
    """Features -> residual (obs - aurora). The features are already standardised
    (mean/std computed ONLY on the training data, so that val/test do not leak)."""

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
# Model
# --------------------------------------------------------------------------- #

class Rete(nn.Module):
    """Network of the final model (and of the experiments, through cv_comune.py): small on purpose,
    it is a local correction and must not relearn meteorology from scratch."""

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
    """Previous version of the head (Lightning, 64-64 without dropout, a single network): used only
    by esperimento_sarah_c.py. The final model uses Rete."""

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

NON_NEGATIVI = {"rr7", "qq2"}  # precipitation and radiation: the corrected value cannot be < 0


def allena_rete(tr, feature, epoche, seed, mean, std, hidden=HIDDEN, dropout=DROPOUT, lr=LR, wd=WEIGHT_DECAY,
                batch=BATCH):
    """Trains a network on the residual obs - aurora. Same training loop as the cross-validation
    (cv_comune.allena_valuta): what was validated is exactly what is trained."""
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
    """Corrected value with the ensemble: mean of the networks (each clipped at zero for
    precipitation and radiation, as in the cross-validation). modello = dictionary saved by main()."""
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
    return torch.load(checkpoint(param), weights_only=False)


def evaluate(modello, df, param, label):
    pred, obs, aur = prevedi(modello, df, param), df["obs"].values, df["aurora"].values
    print(f"[{label}] n={len(df)}")
    print(f"  interpolated Aurora -> RMSE={np.sqrt(np.mean((aur - obs) ** 2)):.3f}  bias={np.mean(aur - obs):.3f}")
    print(f"  downscaling         -> RMSE={np.sqrt(np.mean((pred - obs) ** 2)):.3f}  bias={np.mean(pred - obs):.3f}")


def main():
    """Final model: for each variable trains N_SEED networks on ALL the training rows (all years,
    all regions, Piedmont excluded), for a fixed number of epochs chosen by the region x year
    cross-validation (cv_regione_anno.py -> epoche_cv.json), without early stopping.
    Saves the networks and the feature normalisation; evaluates the test set (Piedmont) only with
    --valuta_test."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--parameter", choices=PARAMS, default=None, help="a single variable (default: all)")
    ap.add_argument("--table", default=os.path.splitext(TABELLA_TRAINING)[0],
                    help="basename (without extension) of the training table")
    ap.add_argument("--test_table", default=os.path.splitext(TABELLA_TEST)[0],
                    help="basename (without extension) of the test table (Piedmont)")
    ap.add_argument("--valuta_test", action="store_true",
                    help="also evaluate on the Piedmont test set: use it ONCE only, when the model is final")
    ap.add_argument("--senza_esclusioni", action="store_true",
                    help="do not apply esclusioni_training.csv (for comparison)")
    args = ap.parse_args()
    torch.set_num_threads(6)
    if not os.path.exists(EPOCHE_CV):
        raise SystemExit("cannot find epoche_cv.json: run cv_regione_anno.py first")
    epoche = json.load(open(EPOCHE_CV))
    os.makedirs(CHECKPOINTS, exist_ok=True)

    for param in ([args.parameter] if args.parameter else PARAMS):
        feature = feature_cols(param)
        df = load_tables(param, table=args.table, test_table=args.test_table,
                         usa_esclusioni=not args.senza_esclusioni)
        mancanti = [c for c in feature if c not in df.columns]
        if mancanti:
            raise SystemExit(f"the table is missing {mancanti}: re-run the notebook §6.6")
        train_df = df[df["split"] != "test"].dropna(subset=feature + ["obs", "aurora"])
        test_df = df[df["split"] == "test"].dropna(subset=feature + ["obs", "aurora"])
        print(f"\n=== {param}: train={len(train_df)}  test(Piedmont)={len(test_df)}  epochs={epoche[param]}  "
              f"features={len(feature)}  networks={N_SEED} ===")
        mean = train_df[feature].values.mean(axis=0)
        std = train_df[feature].values.std(axis=0)
        std[std == 0] = 1.0
        reti = [allena_rete(train_df, feature, epoche[param], seed, mean, std).state_dict() for seed in range(N_SEED)]
        modello = {"feature_cols": feature, "mean": mean.tolist(), "std": std.tolist(), "epoche": epoche[param],
                   "hidden": list(HIDDEN), "dropout": DROPOUT, "reti": reti}
        torch.save(modello, checkpoint(param))
        print(f"saved {checkpoint(param)}")
        evaluate(modello, train_df, param, "train (data seen in training)")
        if not args.valuta_test:
            print("[test] not evaluated: add --valuta_test once only, when the model is final.")
        elif len(test_df):
            evaluate(modello, test_df, param, "test (Piedmont 2020, region and year never seen)")
        else:
            print("[test] no test rows available - skipped.")


if __name__ == "__main__":
    main()
