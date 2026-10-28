"""
Shared parts of the model experiments (simple baselines, terrain descriptors, precipitation,
hyperparameters, ensemble): same table, same exclusions, same folds and same error measure as
cv_regione_anno.py, so the variants differ only in what is being tested.

Region x year cross-validation: for each region R (Lombardia, Emilia-Romagna, Veneto) and year Y
(2016, 2022, 2023) the model is trained on the other regions in the other years and evaluated on R
in year Y. Piedmont (test) is never used.
"""
import geopandas as gpd
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from shapely.geometry import Point

from train_downscaling_head import FEATURE_COLS, Rete, applica_esclusioni  # Rete: the same as the final model

# Paths relative to the main project folder (where the script is run from)
TABLE = "tabelle/training_table_aurora.parquet"  # notebook §6.6
REGIONI = "dati/statici/italy_regions.geojson"
REGIONS = ["Lombardia", "Emilia-Romagna", "Veneto"]  # validation regions (Piedmont is the test)
PARAMS = ["tx3", "tn3", "rr7", "qq2"]
BATCH = 4096
SEED = 0
NON_NEGATIVI = {"rr7", "qq2"}


def assign_regions(df):
    """Adds the column "regione" (administrative region of the station, italy_regions.geojson)."""
    st = df.drop_duplicates("uid")[["uid", "lat", "lon"]]
    regions = gpd.read_file(REGIONI)
    name_col = [c for c in regions.columns if "name" in c.lower()][0]
    g = gpd.GeoDataFrame(st, geometry=[Point(x, y) for x, y in zip(st.lon, st.lat)], crs="EPSG:4326")
    j = gpd.sjoin(g, regions[[name_col, "geometry"]], how="left", predicate="within")
    reg_map = j.drop_duplicates("uid").set_index("uid")[name_col]
    return df.assign(regione=df["uid"].map(reg_map))


def make_tensors(d, mean, std):
    """Standardised features, residual, Aurora and observation as tensors (used by cv_regione_anno.py)."""
    x = torch.tensor((d[FEATURE_COLS].values - mean) / std, dtype=torch.float32)
    y = torch.tensor(d["residual"].values, dtype=torch.float32).unsqueeze(1)
    aurora = torch.tensor(d["aurora"].values, dtype=torch.float32).unsqueeze(1)
    obs = torch.tensor(d["obs"].values, dtype=torch.float32).unsqueeze(1)
    return x, y, aurora, obs


def carica_tabella(colonne_extra=None):
    """Training table (without Piedmont, with exclusions), with season, year and region."""
    t = pd.read_parquet(TABLE)
    t = applica_esclusioni(t[t["split"] != "test"])
    doy = t["date"].dt.dayofyear.astype(float)
    t["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    t["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    t["anno"] = t["date"].dt.year
    t["mese"] = t["date"].dt.month
    t = assign_regions(t)
    if colonne_extra is not None:
        t = t.merge(colonne_extra, on="uid", how="left")
    return t.dropna(subset=["residual"] + FEATURE_COLS).reset_index(drop=True)


def folds(tp):
    """(region, year, training rows, validation rows) for the 9 folds."""
    for reg in REGIONS:
        for anno in sorted(tp["anno"].unique()):
            tr = tp[(tp["regione"] != reg) & (tp["anno"] != anno)]
            va = tp[(tp["regione"] == reg) & (tp["anno"] == anno)]
            if len(tr) and len(va):
                yield reg, anno, tr, va


def errori(pred, obs):
    e = pred - obs
    return {"rmse": float(np.sqrt(np.mean(e ** 2))), "bias": float(np.mean(e)), "mae": float(np.mean(np.abs(e)))}


# Target transformations for precipitation: the network learns f(obs) - f(aurora) and the forecast is
# finv(f(aurora) + output). "nessuna" (none) = plain residual, as in the current model.
TRASFORMAZIONI = {
    "nessuna": (lambda x: x, lambda x: x),
    "radice": (lambda x: np.sqrt(np.clip(x, 0, None)), lambda x: np.clip(x, 0, None) ** 2),
    "log1p": (lambda x: np.log1p(np.clip(x, 0, None)), lambda x: np.expm1(np.clip(x, 0, None))),
}


def allena_valuta(tr, va, feature, param, hidden=(64, 64), dropout=0.0, lr=1e-3, wd=1e-4,
                  max_epoche=40, seed=SEED, trasformazione="nessuna", batch=BATCH, salva_pred=False):
    """Trains a network on tr and returns the error on va after each epoch (original scale).
    With salva_pred=True it also returns the predictions of each epoch (for the ensemble)."""
    torch.manual_seed(seed)
    f, finv = TRASFORMAZIONI[trasformazione]
    m = tr[feature].values.mean(axis=0)
    s = tr[feature].values.std(axis=0)
    s[s == 0] = 1
    x = torch.tensor((tr[feature].values - m) / s, dtype=torch.float32)
    y = torch.tensor(f(tr["obs"].values) - f(tr["aurora"].values), dtype=torch.float32).unsqueeze(1)
    xv = torch.tensor((va[feature].values - m) / s, dtype=torch.float32)
    av, ov = va["aurora"].values, va["obs"].values
    net = Rete(len(feature), hidden, dropout)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    curva, pred = [], []
    for ep in range(1, max_epoche + 1):
        net.train()
        perm = torch.randperm(len(y))
        for i in range(0, len(y), batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            torch.nn.functional.mse_loss(net(x[b]), y[b]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            p = finv(f(av) + net(xv).numpy().ravel())
        if param in NON_NEGATIVI:
            p = np.clip(p, 0, None)
        curva.append({"epoca": ep, **errori(p, ov)})
        if salva_pred:
            pred.append(p.astype("float32"))
    return (curva, np.array(pred)) if salva_pred else curva


def cv_rete(t, param, feature, etichetta, **opzioni):
    """Region x year CV of one configuration: one row per fold and epoch."""
    righe = []
    for reg, anno, tr, va in folds(t[t["parameter"] == param]):
        for r in allena_valuta(tr, va, feature, param, **opzioni):
            righe.append({"variante": etichetta, "parametro": param, "regione": reg, "anno": anno, **r})
    return righe


def riepilogo(res, chiavi=("variante", "parametro")):
    """Mean RMSE over the folds at the best epoch (chosen on the mean over folds) for each configuration."""
    chiavi = list(chiavi)
    curva = res.groupby(chiavi + ["epoca"])[["rmse", "bias", "mae"]].mean().reset_index()
    best = curva.loc[curva.groupby(chiavi)["rmse"].idxmin()].set_index(chiavi)
    return best[["epoca", "rmse", "bias", "mae"]]


RIFERIMENTO_ATTUALE = {"tx3": 2.347, "tn3": 2.322, "rr7": 6.454, "qq2": 34.653}  # cv_regione_anno.py
