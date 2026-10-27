"""
Lettura dei campi giornalieri di Aurora salvati da extract_aurora_campi.py (aurora_campi/, un file per
giorno) e interpolazione nei punti voluti (stazioni o pixel di una griglia fine).

    campi = carica_campi([2016, 2022, 2023])          # Dataset: date x lat x lon (+ livello)
    punti = interpola_punti(campi, lat, lon)          # DataFrame: una riga per punto e giorno

Le variabili in quota (atm_<nome>_media, dimensione "livello") diventano una colonna per livello
(atm_t_850, atm_q_700, ...). Le variabili statiche (orog_aurora, statica_*) non dipendono dal giorno.
"""
import glob
import os
import re

import numpy as np
import pandas as pd
import xarray as xr

CARTELLA = "dati/aurora_campi"  # extract_aurora_campi.py
_RE = re.compile(r"aurora_campi_(\d{4})-(\d{2})-(\d{2})\.nc$")


def _statica(nome):
    return nome == "orog_aurora" or nome.startswith("statica_")


def carica_campi(anni, variabili=None):
    """Campi giornalieri degli anni richiesti. variabili: lista di nomi da tenere (None = tutte)."""
    files = []
    for f in sorted(glob.glob(os.path.join(CARTELLA, "aurora_campi_*.nc"))):
        m = _RE.search(os.path.basename(f))
        if m and int(m.group(1)) in set(anni):
            files.append((pd.Timestamp(f"{m.group(1)}-{m.group(2)}-{m.group(3)}"), f))
    if not files:
        raise FileNotFoundError(f"nessun file in {CARTELLA}/ per gli anni {anni}")
    # lettura con netCDF4 invece di xr.open_dataset: ~100 volte piu' veloce su tanti file piccoli
    import netCDF4

    with netCDF4.Dataset(files[0][1]) as d:
        lat, lon = d["lat"][:].data, d["lon"][:].data
        livelli = d["livello"][:].data if "livello" in d.variables else None
        nomi = [v for v in d.variables if v not in ("lat", "lon", "livello")
                and (variabili is None or v in variabili)]
        dims = {v: d[v].dimensions for v in nomi}
        statiche = {v: np.asarray(d[v][:]) for v in nomi if _statica(v)}
    dinamiche = [v for v in nomi if not _statica(v)]
    dati = {v: [] for v in dinamiche}
    for _, f in files:
        with netCDF4.Dataset(f) as d:
            for v in dinamiche:
                dati[v].append(np.asarray(d[v][:]))
    coords = {"date": [g for g, _ in files], "lat": lat, "lon": lon}
    if livelli is not None:
        coords["livello"] = livelli
    ds = {v: (("date",) + dims[v], np.stack(dati[v])) for v in dinamiche}
    ds.update({v: (dims[v], a) for v, a in statiche.items()})
    return xr.Dataset(ds, coords=coords)


def _idw(campi, lat, lon, k, potenza):
    """Media pesata 1/d**potenza sui k centri di cella piu' vicini a ogni punto (distanza sulla sfera)."""
    from scipy.spatial import cKDTree

    def xyz(la, lo):
        la, lo = np.radians(la), np.radians(lo)
        return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)

    glat, glon = np.meshgrid(campi["lat"].values, campi["lon"].values, indexing="ij")
    d, idx = cKDTree(xyz(glat.ravel(), glon.ravel())).query(xyz(lat, lon), k=k)
    d, idx = d.reshape(len(lat), k), idx.reshape(len(lat), k)
    w = 1.0 / np.maximum(d, 1e-12) ** potenza  # punto sul centro di una cella: prende quel valore
    w /= w.sum(axis=1, keepdims=True)
    piatto = campi.stack(cella=("lat", "lon")).drop_vars(["cella", "lat", "lon"])
    vicini = piatto.isel(cella=xr.DataArray(idx, dims=("punto", "vicino")))
    return (vicini * xr.DataArray(w, dims=("punto", "vicino"))).sum("vicino")


# da passare come metodi=... per interpolare pioggia e radiazione con IDW e il resto con `metodo`
IDW_PIOGGIA_RADIAZIONE = {v: "idw" for v in ("rr7_aurora_raw", "qq2_aurora_raw",
                                              "sup_scaled_tp_1h_media", "sup_ssrd_1h_media")}


def interpola_punti(campi, lat, lon, metodo="linear", ids=None, metodi=None, k=4, potenza=2):
    """Interpola tutti i campi nei punti (lat, lon). metodo: "linear" (bilineare), "nearest" o "idw"
    (inverso della distanza sui k centri di cella piu' vicini, pesi 1/d**potenza).
    metodi: {variabile: metodo} per usare su alcune variabili un metodo diverso da `metodo`
    (es. metodi=IDW_PIOGGIA_RADIAZIONE).
    Restituisce un DataFrame con colonne punto (o ids), date e una colonna per variabile
    (le variabili in quota diventano una colonna per livello)."""
    lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    pts = dict(lat=xr.DataArray(lat, dims="punto"), lon=xr.DataArray(lon, dims="punto"))
    scelta = {n: (metodi or {}).get(n, metodo) for n in campi.data_vars}
    parti = []
    for m in dict.fromkeys(scelta.values()):
        sotto = campi[[n for n, s in scelta.items() if s == m]]
        parti.append(_idw(sotto, lat, lon, k, potenza) if m == "idw" else sotto.interp(**pts, method=m))
    v = xr.merge(parti)[list(campi.data_vars)]
    colonne = {}
    for nome, da in v.data_vars.items():
        if "livello" in da.dims:
            for lv in da["livello"].values:
                colonne[f"{nome.replace('_media', '')}_{int(lv)}"] = da.sel(livello=lv).drop_vars("livello")
        else:
            colonne[nome] = da
    df = xr.Dataset(colonne).to_dataframe().reset_index()
    df = df.drop(columns=[c for c in ["lat", "lon", "livello"] if c in df.columns])
    if ids is not None:
        df["punto"] = np.asarray(ids)[df["punto"].values]
    return df
