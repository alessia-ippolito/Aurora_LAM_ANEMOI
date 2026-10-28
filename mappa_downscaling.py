"""
High-resolution maps (~1 km) over the AOI: applies the final model (train_downscaling_head.py: one
ensemble of networks per variable) in each pixel, with the same inputs as in training (notebook §6.6).

For each day:
  - grid of ~0.01 degrees over the AOI, elevation = GLO-30 DEM at the pixel centre (at the point, as
    for the stations: each pixel is the expected value at a point in the centre of the cell);
  - daily Aurora fields (aurora_campi/, extract_aurora_campi.py) bilinearly interpolated in each
    pixel: value of the variable and all the other variables used as features (AURORA_FEATURE in
    fattori_scala_aurora.json), pixel elevation - Aurora elevation;
  - terrain shape at the pixel (descrittori_terreno.py, as for the stations), used for tn3 and qq2;
  - model: corrected value = Aurora + mean of the network corrections; precipitation and radiation
    clipped at zero;
  - comparison with the stations of that day (ECA&D, ARPA Lombardia, ARPA Piemonte, ARPAE), same
    rules as the notebook (§6.6): RMSE and bias of interpolated Aurora and of the model.

The field files must contain all the variables (version of extract_aurora_campi.py with sup_*/atm_*).

Usage:
    python mappa_downscaling.py 2021-08-15 2021-10-04 2021-12-31
Output:
    mappa_<day>.nc, mappa_<day>.png, mappa_confronto_stazioni.csv
"""
import glob
import json
import os
import sys

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from pyproj import Transformer

from campi_aurora import carica_campi, interpola_punti
from descrittori_terreno import FORMA_RILIEVO, dem_elevation, descrittori
from train_downscaling_head import carica_modello, prevedi

# Paths relative to the main project folder (where the script is run from)
CACHE_ECAD = "dati/osservazioni/ecad_padania_raw_cache"
CACHE_LOMBARDIA = "dati/osservazioni/arpa_lombardia_daily_cache_utc"
CACHE_PIEMONTE = "dati/osservazioni/arpa_piemonte_daily_cache"
CACHE_ARPAE = "dati/osservazioni/arpae_qq2_daily_cache"
STAZIONI_ECAD = "dati/osservazioni/stazioni_padania.csv"
SENSORI_LOMBARDIA = "dati/osservazioni/lombardia_sensori_target.csv"
STAZIONI_PIEMONTE = "dati/osservazioni/arpa_piemonte_stazioni.csv"
CAMPI_AURORA = "dati/aurora_campi"
DEM = "dati/statici/cop30_padania.tif"
REGIONI = "dati/statici/italy_regions.geojson"
FATTORI_SCALA = "tabelle/fattori_scala_aurora.json"

GIORNI = sys.argv[1:] or ["2021-08-15", "2021-10-04", "2021-12-31"]
PARAMS = ["tx3", "tn3", "qq2", "rr7"]
COL_AURORA = {"tx3": "tx3_aurora", "tn3": "tn3_aurora", "rr7": "rr7_aurora_raw", "qq2": "qq2_aurora_raw"}
RIS = 0.01  # degrees, ~1 km
UNITA = {"tx3": "°C", "tn3": "°C", "qq2": "W/m²", "rr7": "mm"}
INFO = json.load(open(FATTORI_SCALA))  # written by the notebook §6.6
FATTORI, TABLE_QC = INFO["fattori_scala"], {k: tuple(v) for k, v in INFO["table_qc"].items()}


def carica_teste():
    """Final model for each variable (checkpoints/<p>_finale.pt, train_downscaling_head.py)."""
    return {p: carica_modello(p) for p in PARAMS}


def feature_punti(campi_giorno, lat, lon, elev):
    """All the Aurora variables bilinearly interpolated at the points, as in the notebook §6.6."""
    f = interpola_punti(campi_giorno, lat, lon, "linear").sort_values("punto").reset_index(drop=True)
    for p, c in COL_AURORA.items():
        f[f"a_{p}"] = (f[c] * FATTORI[p]).clip(*TABLE_QC[p])
    f["elev"] = np.asarray(elev)
    f["dz"] = f["elev"] - f["orog_aurora"]
    forma = descrittori(np.asarray(lat), np.asarray(lon), con_distanze=False)[FORMA_RILIEVO]
    return pd.concat([f, forma.set_axis(f.index)], axis=1)


def correggi(modello, f, param, doy):
    """Applies the model to the features f (DataFrame, one row per point); returns the corrected value."""
    f = f.assign(aurora=f[f"a_{param}"], doy_sin=np.sin(2 * np.pi * doy / 365.25),
                 doy_cos=np.cos(2 * np.pi * doy / 365.25))
    return prevedi(modello, f, param)


def griglia_dem():
    """Grid of ~RIS degrees over the DEM extent, elevation = 30 m DEM at the pixel centre.
    At the point, not the pixel mean: as for the stations (dem_elevation) and as the terrain shape,
    so each pixel is the expected value at a point in the centre of the cell, with features of the
    same kind as in training."""
    with rasterio.open(DEM) as d:
        b = d.bounds
    ny, nx = int(round((b.top - b.bottom) / RIS)), int(round((b.right - b.left) / RIS))
    lat = b.top - (np.arange(ny) + 0.5) * (b.top - b.bottom) / ny
    lon = b.left + (np.arange(nx) + 0.5) * (b.right - b.left) / nx
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    elev = dem_elevation(LAT.ravel(), LON.ravel()).reshape(ny, nx)
    return lat, lon, elev


def osservazioni(giorni):
    """Observations of the requested days, with the same rules as the notebook §6.6."""
    giorni = set(pd.to_datetime(giorni))
    out = []
    anag = pd.read_csv(STAZIONI_ECAD).set_index("station_id")
    for f in glob.glob(os.path.join(CACHE_ECAD, "*.csv")):
        sid, p = os.path.basename(f)[:-4].split("__")
        if p not in PARAMS or sid not in anag.index:
            continue
        d = pd.read_csv(f)
        if d.empty:
            continue
        d["date"] = pd.to_datetime(d["date"], utc=True).dt.tz_convert(None) - pd.Timedelta(days=1)
        d = d[d["date"].isin(giorni)].dropna(subset=[p])
        out += [(sid, "ecad", p, r.date, getattr(r, p), anag.loc[sid, "lat"], anag.loc[sid, "lon"])
                for r in d.itertuples()]
    reg = pd.read_csv(SENSORI_LOMBARDIA).set_index("idsensore")
    reg["lon_utm"], reg["lat_utm"] = Transformer.from_crs("EPSG:32632", "EPSG:4326", always_xy=True).transform(
        reg["cgb_est"].values, reg["cgb_nord"].values)
    col = {"Temperatura": [("tx3", "vmax"), ("tn3", "vmin")], "Precipitazione": [("rr7", "vsum")],
           "Radiazione Globale": [("qq2", "vmed")]}
    for f in glob.glob(os.path.join(CACHE_LOMBARDIA, "*.csv")):
        s = int(os.path.basename(f)[:-4])
        d = pd.read_csv(f)
        if d.empty or s not in reg.index:
            continue
        d["giorno"] = pd.to_datetime(d["giorno"])
        d = d.sort_values("giorno")
        attese = np.where(d["n"].rolling(31, center=True, min_periods=1).median() <= 30, 24, 144)
        d = d[(d["n"] >= 0.8 * attese) & d["giorno"].isin(giorni)]
        for p, c in col[reg.loc[s, "tipologia"]]:
            out += [(f"arpalom_{s}", "arpa_lom", p, r.giorno, getattr(r, c), reg.loc[s, "lat_utm"], reg.loc[s, "lon_utm"])
                    for r in d.itertuples()]
    pie = pd.read_csv(STAZIONI_PIEMONTE).set_index("uid")
    pcol = {"tmax": ("tx3", 1.0), "tmin": ("tn3", 1.0), "ptot": ("rr7", 1.0), "rtot": ("qq2", 1e6 / 86400)}
    for f in glob.glob(os.path.join(CACHE_PIEMONTE, "*.csv")):
        u = os.path.basename(f)[:-4]
        if u not in pie.index:
            continue
        d = pd.read_csv(f, parse_dates=["data"])
        d = d[d["data"].isin(giorni)]
        for c, (p, k) in pcol.items():
            out += [(u, "arpa_piemonte", p, r.data, getattr(r, c) * k, pie.loc[u, "lat"], pie.loc[u, "lon"])
                    for r in d.dropna(subset=[c]).itertuples()]
    # ARPAE (Emilia-Romagna radiation, notebook §6.3): monthly files already daily; the last day of a
    # month also appears in the file of the next month, so the hours are recombined before the filter
    mesi = {(g.year, g.month) for g in giorni} | {((g + pd.offsets.MonthBegin(1)).year,
                                                   (g + pd.offsets.MonthBegin(1)).month) for g in giorni}
    fa = [os.path.join(CACHE_ARPAE, f"{y}-{m:02d}.csv") for y, m in sorted(mesi)]
    fa = [f for f in fa if os.path.exists(f)]
    if fa:
        a = pd.concat([pd.read_csv(f, parse_dates=["date"]) for f in fa], ignore_index=True)
        a["somma"] = a["qq2"] * a["n_ore"]
        a = a.groupby(["nome", "lat", "lon", "date"], dropna=False)[["somma", "n_ore"]].sum().reset_index()
        a = a[(a["n_ore"] >= 20) & a["date"].isin(giorni)]
        out += [("arpae_" + r.nome.replace(" ", "_"), "arpae", "qq2", r.date, r.somma / r.n_ore, r.lat, r.lon)
                for r in a.itertuples()]
    o = pd.DataFrame(out, columns=["uid", "source", "parameter", "date", "obs", "lat", "lon"])
    lo = o["parameter"].map({"tx3": -30, "tn3": -40, "qq2": 0, "rr7": 0})
    hi = o["parameter"].map({"tx3": 50, "tn3": 35, "qq2": 450, "rr7": 800})
    return o[o["obs"].between(lo, hi)]


def _file_campi(giorno):
    return os.path.join(CAMPI_AURORA, f"aurora_campi_{giorno}.nc")


def main():
    print("scale factors:", FATTORI)
    teste = carica_teste()
    glat, glon, gelev = griglia_dem()
    LAT, LON = np.meshgrid(glat, glon, indexing="ij")
    regioni = gpd.read_file(REGIONI)
    obs = osservazioni(GIORNI)
    confronto = []

    for giorno in GIORNI:
        day = pd.Timestamp(giorno)
        campi = carica_campi([day.year])
        if np.datetime64(day) not in campi["date"].values:
            print(f"skipping {day.date()}: {_file_campi(day.date())} is missing")
            continue
        campi = campi.sel(date=[day])
        mancanti = [c for c in teste["tx3"]["feature_cols"]
                    if c.startswith(("sup_", "atm_")) and c.rsplit("_", 1)[0] + "_media" not in campi
                    and c not in campi]
        if mancanti:
            print(f"skipping {day.date()}: the field file does not have all the variables ({mancanti[:3]}...): "
                  f"it must be re-extracted with the current version of extract_aurora_campi.py")
            continue
        doy = day.dayofyear
        f_griglia = feature_punti(campi, LAT.ravel(), LON.ravel(), gelev.ravel())
        out = xr.Dataset(coords={"lat": glat, "lon": glon})
        out["elev"] = (("lat", "lon"), gelev)
        fig, axes = plt.subplots(len(PARAMS), 3, figsize=(17, 4.2 * len(PARAMS)))
        for r, p in enumerate(PARAMS):
            modello = teste[p]
            campo = (campi[COL_AURORA[p]].isel(date=0) * FATTORI[p]).clip(*TABLE_QC[p])
            a25 = campo.interp(lat=glat, lon=glon, method="nearest").values           # Aurora 0.25 degrees
            a1 = f_griglia[f"a_{p}"].values.reshape(LAT.shape)                         # interpolated Aurora
            c1 = correggi(modello, f_griglia, p, doy).reshape(LAT.shape)               # downscaling
            out[f"{p}_aurora_25km"] = (("lat", "lon"), a25)
            out[f"{p}_aurora_bilineare"] = (("lat", "lon"), a1)
            out[f"{p}_downscaling"] = (("lat", "lon"), c1)

            # comparison with the stations of that day
            o = obs[(obs["parameter"] == p) & (obs["date"] == day)].copy()
            if len(o):
                el = dem_elevation(o["lat"].values, o["lon"].values)
                o = o[~np.isnan(el)]
                el = el[~np.isnan(el)]
                fo = feature_punti(campi, o["lat"].values, o["lon"].values, el)
                ok = fo[[c for c in modello["feature_cols"] if c not in ("aurora", "doy_sin", "doy_cos")]
                        ].notna().all(axis=1).values
                o, fo = o[ok], fo[ok]
                for nome, v in {"Aurora (bilinear)": fo[f"a_{p}"].values,
                                "downscaling": correggi(modello, fo, p, doy)}.items():
                    e = v - o["obs"].values
                    confronto.append({"giorno": str(day.date()), "parametro": p, "versione": nome, "n": len(e),
                                      "rmse": float(np.sqrt(np.mean(e ** 2))), "bias": float(e.mean())})

            # figure: Aurora 25 km | interpolated Aurora | downscaling ~1 km, stations overlaid
            vmin, vmax = np.nanpercentile(np.concatenate([a25.ravel(), c1.ravel()]), [2, 98])
            cmap = {"tx3": "RdYlBu_r", "tn3": "RdYlBu_r", "qq2": "magma", "rr7": "Blues"}[p]
            ext = [glon.min(), glon.max(), glat.min(), glat.max()]
            for c, (titolo, m) in enumerate([("Aurora 0.25°", a25), ("Aurora interpolated (bilinear)", a1),
                                              ("Downscaling ~1 km", c1)]):
                ax = axes[r, c]
                im = ax.imshow(m, extent=ext, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax,
                               aspect=1 / np.cos(np.radians(45)), interpolation="nearest")
                regioni.boundary.plot(ax=ax, color="k", linewidth=0.4, aspect=None)
                if c == 0:
                    # Aurora grid points: each cell is the area closest to one point
                    plon, plat = np.meshgrid(campo["lon"].values, campo["lat"].values)
                    ax.scatter(plon, plat, s=3, color="k", linewidth=0)
                if c == 2 and len(o):
                    ax.scatter(o["lon"], o["lat"], c=o["obs"], cmap=cmap, vmin=vmin, vmax=vmax, s=12,
                               edgecolor="k", linewidth=0.3)
                ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
                ax.set_title(f"{p} - {titolo}", fontsize=9)
            fig.colorbar(im, ax=axes[r, :], shrink=0.8, label=f"{p} ({UNITA[p]})")
        fig.suptitle(f"{day.date()}: Aurora 25 km and downscaling ~1 km (left: Aurora grid points; "
                     f"right: observed stations)", fontsize=13)
        fig.savefig(f"mappe/mappa_{day.date()}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        out.to_netcdf(f"mappe/mappa_{day.date()}.nc")
        print(f"saved mappa_{day.date()}.png and mappa_{day.date()}.nc")

    if confronto:
        c = pd.DataFrame(confronto)
        c.to_csv("risultati/mappa_confronto_stazioni.csv", index=False)
        print("\ncomparison with the stations (RMSE):")
        print(c.pivot_table(index=["giorno", "parametro"], columns="versione", values="rmse").round(2))


if __name__ == "__main__":
    main()
