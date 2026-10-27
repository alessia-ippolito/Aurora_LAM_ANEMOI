"""
Esperimento: per la radiazione (qq2) conviene ancora il pre-training sui pixel SARAH-3, ora che la
testa riceve tutte le variabili di Aurora (notebook §6.6)?

Pixel SARAH come osservazioni qq2 aggiuntive SOLO in training (mai come feature: in inferenza la
testa usa solo Aurora + feature statiche). Per ogni pixel, Aurora e tutte le variabili sono
interpolate bilinearmente dai campi (aurora_campi/), esattamente come per le stazioni; quota dal DEM.

Cross-validation regione x anno (come cv_regione_anno.py): si valuta SOLO sulle stazioni vere della
regione e dell'anno tenuti fuori. Dal pre-training si escludono i pixel della regione E dell'anno
tenuti fuori: SARAH e' un'osservazione satellitare di quel giorno, includere l'anno di validazione
darebbe alla rete il meteo reale di quei giorni e una stima ottimista.

Varianti (2 seed ciascuna):
  STAZIONI = solo stazioni (= modello attuale)
  PRE+FT   = pre-training sui pixel SARAH, poi fine-tuning sulle sole stazioni
L'errore si misura dopo ogni epoca di fine-tuning: per ogni variante si sceglie l'epoca migliore in
media sui fold.

Uso:
    python esperimento_sarah_c.py
Output:
    sarah_c_pixel_<anni>.parquet (cache dei pixel), esperimento_sarah_c_risultati.csv
"""
import glob
import json
import os
import re

import netCDF4
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from campi_aurora import carica_campi, interpola_punti
from descrittori_terreno import dem_elevation
from cv_comune import BATCH, REGIONS, assign_regions
from train_downscaling_head import FEATURE_COLS, DownscalingHead, applica_esclusioni

TABLE = "tabelle/training_table_aurora.parquet"
INFO = json.load(open("tabelle/fattori_scala_aurora.json"))
SARAH_RAW = "dati/statici/sarah3_padania_raw"
PASSO_PIXEL = 2          # un pixel SARAH ogni 2 (0.1 gradi, ~10 km)
EPOCHE_PRE = 20
MAX_EPOCHS_FT = 40
SEEDS = [0, 1]
COL_AURORA = {"tx3": "tx3_aurora", "tn3": "tn3_aurora", "rr7": "rr7_aurora_raw", "qq2": "qq2_aurora_raw"}


def pixel_sarah(anni):
    """Pixel SARAH (uno ogni PASSO_PIXEL) dentro le regioni di validazione, con SIS giornaliero e
    tutte le feature di Aurora interpolate nel punto. Cache su disco."""
    cache = f"tabelle/sarah_c_pixel_{'_'.join(map(str, anni))}.parquet"
    if os.path.exists(cache):
        print(f"pixel SARAH da cache: {cache}")
        return pd.read_parquet(cache)
    files = sorted(f for f in glob.glob(os.path.join(SARAH_RAW, "SISdm*.nc"))
                   if int(re.search(r"SISdm(\d{4})", f).group(1)) in anni)
    with netCDF4.Dataset(files[0]) as d:
        lat, lon = d["lat"][:].data, d["lon"][:].data
    iy = np.where((lat >= 44.0) & (lat <= 46.5))[0][::PASSO_PIXEL]
    ix = np.where((lon >= 6.5) & (lon <= 13.0))[0][::PASSO_PIXEL]
    LA, LO = np.meshgrid(lat[iy], lon[ix], indexing="ij")
    px = pd.DataFrame({"uid": [f"sarah_{i}" for i in range(LA.size)], "lat": LA.ravel(), "lon": LO.ravel()})
    px = assign_regions(px)
    px = px[px["regione"].isin(REGIONS)].reset_index(drop=True)
    px["elev"] = dem_elevation(px["lat"].values, px["lon"].values)
    px = px.dropna(subset=["elev"]).reset_index(drop=True)
    sel_y = np.searchsorted(iy, [int(np.where(lat == v)[0][0]) for v in px["lat"]])
    sel_x = np.searchsorted(ix, [int(np.where(lon == v)[0][0]) for v in px["lon"]])
    print(f"pixel SARAH nelle regioni di validazione: {len(px)}; lettura di {len(files)} file...", flush=True)
    righe = []
    for f in files:
        giorno = pd.Timestamp(re.search(r"SISdm(\d{8})", f).group(1))
        with netCDF4.Dataset(f) as d:
            # SIS e' salvato come intero: in float prima di mettere NaN nei valori mancanti
            blocco = np.ma.masked_array(d["SIS"][0, iy[0]:iy[-1] + 1:PASSO_PIXEL,
                                                 ix[0]:ix[-1] + 1:PASSO_PIXEL]).astype(float).filled(np.nan)
        righe.append(pd.DataFrame({"uid": px["uid"], "date": giorno, "obs": blocco[sel_y, sel_x]}))
    s = pd.concat(righe, ignore_index=True).dropna(subset=["obs"])

    # Aurora e tutte le variabili nel punto del pixel, come per le stazioni (notebook §6.6)
    campi = carica_campi(anni)
    f = interpola_punti(campi, px["lat"], px["lon"], "linear", ids=px["uid"]).rename(columns={"punto": "uid"})
    fatt, qc = INFO["fattori_scala"], {k: tuple(v) for k, v in INFO["table_qc"].items()}
    for p, c in COL_AURORA.items():
        f[f"a_{p}"] = (f[c] * fatt[p]).clip(*qc[p])
    f = f[["uid", "date", "orog_aurora"] + [c for c in INFO["aurora_feature"] if c != "dz"]]
    s = s.merge(f, on=["uid", "date"]).merge(px[["uid", "lat", "lon", "elev", "regione"]], on="uid")
    s["dz"] = s["elev"] - s["orog_aurora"]
    s["aurora"] = s["a_qq2"]
    s["residual"] = s["obs"] - s["aurora"]
    s = s.drop(columns="orog_aurora")
    s.to_parquet(cache, index=False)
    print(f"salvato {cache}: {s.shape}")
    return s


def add_doy(df):
    doy = df["date"].dt.dayofyear.astype(float)
    return df.assign(doy_sin=np.sin(2 * np.pi * doy / 365.25), doy_cos=np.cos(2 * np.pi * doy / 365.25),
                     anno=df["date"].dt.year)


def tensori(df, mean, std):
    x = torch.tensor((df[FEATURE_COLS].values - mean) / std, dtype=torch.float32)
    y = torch.tensor(df["residual"].values, dtype=torch.float32).unsqueeze(1)
    return x, y


def allena(net, df, mean, std, epoche, seed, va=None):
    """Allena net su df; se va e' dato, restituisce l'RMSE su va dopo ogni epoca."""
    torch.manual_seed(seed)
    x, y = tensori(df, mean, std)
    loader = DataLoader(TensorDataset(x, y), batch_size=BATCH, shuffle=True)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    curva = []
    if va is not None:
        xv, _ = tensori(va, mean, std)
        av, ov = va["aurora"].values, va["obs"].values
    for ep in range(1, epoche + 1):
        net.train()
        for xb, yb in loader:
            opt.zero_grad()
            torch.nn.functional.mse_loss(net(xb), yb).backward()
            opt.step()
        if va is not None:
            net.eval()
            with torch.no_grad():
                corr = np.clip(av + net(xv).numpy().ravel(), 0, None)
            e = corr - ov
            curva.append({"epoca": ep, "rmse": float(np.sqrt(np.mean(e ** 2))), "bias": float(e.mean()),
                          "rmse_aurora": float(np.sqrt(np.mean((av - ov) ** 2)))})
    return curva


def main():
    torch.set_num_threads(4)
    t = pd.read_parquet(TABLE)
    t = applica_esclusioni(t[(t["split"] == "train") & (t["parameter"] == "qq2")])
    t = assign_regions(add_doy(t))
    anni = sorted(t["anno"].unique())
    sarah = add_doy(pixel_sarah(anni))
    print(f"stazioni qq2: {t['uid'].nunique()} ({len(t)} righe) | pixel SARAH: {sarah['uid'].nunique()} "
          f"({len(sarah)} righe) | bias SARAH - Aurora {sarah['residual'].mean():.1f}, "
          f"stazioni - Aurora {t['residual'].mean():.1f} W/m2", flush=True)

    rows = []
    for reg in REGIONS:
        for anno in anni:
            tr = t[(t["regione"] != reg) & (t["anno"] != anno)]
            va = t[(t["regione"] == reg) & (t["anno"] == anno)]
            pre = sarah[(sarah["regione"] != reg) & (sarah["anno"] != anno)]
            mean = tr[FEATURE_COLS].values.mean(axis=0)
            std = tr[FEATURE_COLS].values.std(axis=0)
            std[std == 0] = 1.0
            for seed in SEEDS:
                torch.manual_seed(seed)
                net = DownscalingHead(n_features=len(FEATURE_COLS)).net
                for r in allena(net, tr, mean, std, MAX_EPOCHS_FT, seed, va):
                    rows.append({"regione": reg, "anno": anno, "seed": seed, "variante": "STAZIONI", **r})
                torch.manual_seed(seed)
                net = DownscalingHead(n_features=len(FEATURE_COLS)).net
                allena(net, pre, mean, std, EPOCHE_PRE, seed)
                for r in allena(net, tr, mean, std, MAX_EPOCHS_FT, seed, va):
                    rows.append({"regione": reg, "anno": anno, "seed": seed, "variante": "PRE+FT", **r})
            print(f"{reg} {anno}: fatto (pre-training su {len(pre)} righe SARAH)", flush=True)
            pd.DataFrame(rows).to_csv("risultati/esperimento_sarah_c_risultati.csv", index=False)

    res = pd.DataFrame(rows)
    curva = res.groupby(["variante", "epoca"])["rmse"].mean()
    best = curva.groupby("variante").agg(["min", "idxmin"])
    best["epoca"] = best["idxmin"].apply(lambda x: x[1])
    print("\n=== qq2, CV regione x anno: RMSE medio sui fold e sui seed, all'epoca migliore ===")
    print(best[["min", "epoca"]].rename(columns={"min": "rmse"}).round(3))
    sel = pd.concat([res[(res["variante"] == v) & (res["epoca"] == e)] for v, e in best["epoca"].items()])
    print(sel.groupby(["regione", "anno", "variante"])["rmse"].mean().unstack("variante").round(2))
    print("\nbias medio:", sel.groupby("variante")["bias"].mean().round(2).to_dict())


if __name__ == "__main__":
    main()
