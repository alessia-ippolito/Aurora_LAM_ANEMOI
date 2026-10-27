"""
Parti comuni degli esperimenti sul modello (riferimenti semplici, descrittori del terreno, pioggia,
iperparametri, ensemble): stessa tabella, stesse esclusioni, stessi fold e stessa misura dell'errore
di cv_regione_anno.py, cosi' le varianti differiscono solo per quello che si sta provando.

Cross-validation regione x anno: per ogni regione R (Lombardia, Emilia-Romagna, Veneto) e anno Y
(2016, 2022, 2023) si allena sulle altre regioni negli altri anni e si valuta su R nell'anno Y.
Il Piemonte (test) non entra mai.
"""
import geopandas as gpd
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from shapely.geometry import Point

from train_downscaling_head import FEATURE_COLS, Rete, applica_esclusioni  # Rete: la stessa del modello finale

# Percorsi relativi alla cartella principale del progetto (da cui si lancia lo script)
TABLE = "tabelle/training_table_aurora.parquet"  # notebook §6.6
REGIONI = "dati/statici/italy_regions.geojson"
REGIONS = ["Lombardia", "Emilia-Romagna", "Veneto"]  # regioni di validazione (il Piemonte e' il test)
PARAMS = ["tx3", "tn3", "rr7", "qq2"]
BATCH = 4096
SEED = 0
NON_NEGATIVI = {"rr7", "qq2"}


def assign_regions(df):
    """Aggiunge la colonna "regione" (regione amministrativa della stazione, italy_regions.geojson)."""
    st = df.drop_duplicates("uid")[["uid", "lat", "lon"]]
    regions = gpd.read_file(REGIONI)
    name_col = [c for c in regions.columns if "name" in c.lower()][0]
    g = gpd.GeoDataFrame(st, geometry=[Point(x, y) for x, y in zip(st.lon, st.lat)], crs="EPSG:4326")
    j = gpd.sjoin(g, regions[[name_col, "geometry"]], how="left", predicate="within")
    reg_map = j.drop_duplicates("uid").set_index("uid")[name_col]
    return df.assign(regione=df["uid"].map(reg_map))


def make_tensors(d, mean, std):
    """Feature standardizzate, residuo, Aurora e osservazione come tensori (usata da cv_regione_anno.py)."""
    x = torch.tensor((d[FEATURE_COLS].values - mean) / std, dtype=torch.float32)
    y = torch.tensor(d["residual"].values, dtype=torch.float32).unsqueeze(1)
    aurora = torch.tensor(d["aurora"].values, dtype=torch.float32).unsqueeze(1)
    obs = torch.tensor(d["obs"].values, dtype=torch.float32).unsqueeze(1)
    return x, y, aurora, obs


def carica_tabella(colonne_extra=None):
    """Tabella di training (senza Piemonte, con esclusioni), con stagione, anno e regione."""
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
    """(regione, anno, righe di training, righe di validazione) per i 9 fold."""
    for reg in REGIONS:
        for anno in sorted(tp["anno"].unique()):
            tr = tp[(tp["regione"] != reg) & (tp["anno"] != anno)]
            va = tp[(tp["regione"] == reg) & (tp["anno"] == anno)]
            if len(tr) and len(va):
                yield reg, anno, tr, va


def errori(pred, obs):
    e = pred - obs
    return {"rmse": float(np.sqrt(np.mean(e ** 2))), "bias": float(np.mean(e)), "mae": float(np.mean(np.abs(e)))}


# Trasformazioni del target per la pioggia: la rete impara f(obs) - f(aurora) e la previsione e'
# finv(f(aurora) + uscita). "nessuna" = residuo semplice, come il modello attuale.
TRASFORMAZIONI = {
    "nessuna": (lambda x: x, lambda x: x),
    "radice": (lambda x: np.sqrt(np.clip(x, 0, None)), lambda x: np.clip(x, 0, None) ** 2),
    "log1p": (lambda x: np.log1p(np.clip(x, 0, None)), lambda x: np.expm1(np.clip(x, 0, None))),
}


def allena_valuta(tr, va, feature, param, hidden=(64, 64), dropout=0.0, lr=1e-3, wd=1e-4,
                  max_epoche=40, seed=SEED, trasformazione="nessuna", batch=BATCH, salva_pred=False):
    """Allena una rete su tr e restituisce l'errore su va dopo ogni epoca (scala originale).
    Con salva_pred=True restituisce anche le previsioni di ogni epoca (per l'ensemble)."""
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
    """CV regione x anno di una configurazione: una riga per fold ed epoca."""
    righe = []
    for reg, anno, tr, va in folds(t[t["parameter"] == param]):
        for r in allena_valuta(tr, va, feature, param, **opzioni):
            righe.append({"variante": etichetta, "parametro": param, "regione": reg, "anno": anno, **r})
    return righe


def riepilogo(res, chiavi=("variante", "parametro")):
    """RMSE medio sui fold all'epoca migliore (scelta in media sui fold) per ogni configurazione."""
    chiavi = list(chiavi)
    curva = res.groupby(chiavi + ["epoca"])[["rmse", "bias", "mae"]].mean().reset_index()
    best = curva.loc[curva.groupby(chiavi)["rmse"].idxmin()].set_index(chiavi)
    return best[["epoca", "rmse", "bias", "mae"]]


RIFERIMENTO_ATTUALE = {"tx3": 2.347, "tn3": 2.322, "rr7": 6.454, "qq2": 34.653}  # cv_regione_anno.py
