"""
Reads the daily Aurora fields saved by extract_aurora_campi.py (aurora_campi/, one file per day)
and interpolates them at the requested points (stations or pixels of a fine grid).

    campi = carica_campi([2016, 2022, 2023])          # Dataset: date x lat x lon (+ livello)
    punti = interpola_punti(campi, lat, lon)          # DataFrame: one row per point and day

Pressure-level variables (atm_<name>_media, dimension "livello") become one column per level
(atm_t_850, atm_q_700, ...). Static variables (orog_aurora, statica_*) do not depend on the day.
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
    """Daily fields of the requested years. variabili: list of names to keep (None = all)."""
    files = []
    for f in sorted(glob.glob(os.path.join(CARTELLA, "aurora_campi_*.nc"))):
        m = _RE.search(os.path.basename(f))
        if m and int(m.group(1)) in set(anni):
            files.append((pd.Timestamp(f"{m.group(1)}-{m.group(2)}-{m.group(3)}"), f))
    if not files:
        raise FileNotFoundError(f"no file in {CARTELLA}/ for the years {anni}")
    # reading with netCDF4 instead of xr.open_dataset: ~100 times faster on many small files
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
    """Weighted mean 1/d**potenza over the k grid-cell centres nearest to each point (distance on the sphere)."""
    from scipy.spatial import cKDTree

    def xyz(la, lo):
        la, lo = np.radians(la), np.radians(lo)
        return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)

    glat, glon = np.meshgrid(campi["lat"].values, campi["lon"].values, indexing="ij")
    d, idx = cKDTree(xyz(glat.ravel(), glon.ravel())).query(xyz(lat, lon), k=k)
    d, idx = d.reshape(len(lat), k), idx.reshape(len(lat), k)
    w = 1.0 / np.maximum(d, 1e-12) ** potenza  # a point on a cell centre takes that value
    w /= w.sum(axis=1, keepdims=True)
    piatto = campi.stack(cella=("lat", "lon")).drop_vars(["cella", "lat", "lon"])
    vicini = piatto.isel(cella=xr.DataArray(idx, dims=("punto", "vicino")))
    return (vicini * xr.DataArray(w, dims=("punto", "vicino"))).sum("vicino")


# pass as metodi=... to interpolate precipitation and radiation with IDW and the rest with `metodo`
IDW_PIOGGIA_RADIAZIONE = {v: "idw" for v in ("rr7_aurora_raw", "qq2_aurora_raw",
                                              "sup_scaled_tp_1h_media", "sup_ssrd_1h_media")}


def interpola_punti(campi, lat, lon, metodo="linear", ids=None, metodi=None, k=4, potenza=2):
    """Interpolates all fields at the points (lat, lon). metodo: "linear" (bilinear), "nearest" or "idw"
    (inverse distance over the k nearest grid-cell centres, weights 1/d**potenza).
    metodi: {variable: method} to use a method other than `metodo` for some variables
    (e.g. metodi=IDW_PIOGGIA_RADIAZIONE).
    Returns a DataFrame with columns punto (or ids), date and one column per variable
    (pressure-level variables become one column per level)."""
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
