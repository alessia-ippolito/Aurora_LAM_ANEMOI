"""
Mappe ad alta risoluzione (~1 km) sull'AOI: applica il modello finale (train_downscaling_head.py:
per ogni variabile un ensemble di reti) in ogni pixel, con gli stessi input del training (notebook §6.6).

Per ogni giorno:
  - griglia ~0.01 gradi sull'AOI, quota = DEM GLO-30 nel centro del pixel (nel punto, come per le
    stazioni: ogni pixel e' il valore atteso in un punto al centro della cella);
  - campi giornalieri di Aurora (aurora_campi/, extract_aurora_campi.py) interpolati
    bilinearmente in ogni pixel: valore della variabile e tutte le altre variabili usate come
    feature (AURORA_FEATURE in fattori_scala_aurora.json), quota del pixel - quota di Aurora;
  - forma del rilievo nel pixel (descrittori_terreno.py, come per le stazioni), usata per tn3 e qq2;
  - modello: valore corretto = Aurora + media delle correzioni delle reti; pioggia e radiazione
    tagliate a zero;
  - confronto con le stazioni di quel giorno (ECA&D, ARPA Lombardia, ARPA Piemonte, ARPAE), stesse
    regole del notebook (§6.6): RMSE e bias di Aurora interpolata e del modello.

I file dei campi devono avere tutte le variabili (versione di extract_aurora_campi.py con sup_*/atm_*).

Uso:
    python mappa_downscaling.py 2021-08-15 2021-10-04 2021-12-31
Output:
    mappa_<giorno>.nc, mappa_<giorno>.png, mappa_confronto_stazioni.csv
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

# Percorsi relativi alla cartella principale del progetto (da cui si lancia lo script)
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
RIS = 0.01  # gradi, ~1 km
UNITA = {"tx3": "°C", "tn3": "°C", "qq2": "W/m²", "rr7": "mm"}
INFO = json.load(open(FATTORI_SCALA))  # scritto dal notebook §6.6
FATTORI, TABLE_QC = INFO["fattori_scala"], {k: tuple(v) for k, v in INFO["table_qc"].items()}


def carica_teste():
    """Modello finale per ogni variabile (checkpoints/<p>_finale.pt, train_downscaling_head.py)."""
    return {p: carica_modello(p) for p in PARAMS}


def feature_punti(campi_giorno, lat, lon, elev):
    """Tutte le variabili di Aurora interpolate bilinearmente nei punti, come nel notebook §6.6."""
    f = interpola_punti(campi_giorno, lat, lon, "linear").sort_values("punto").reset_index(drop=True)
    for p, c in COL_AURORA.items():
        f[f"a_{p}"] = (f[c] * FATTORI[p]).clip(*TABLE_QC[p])
    f["elev"] = np.asarray(elev)
    f["dz"] = f["elev"] - f["orog_aurora"]
    forma = descrittori(np.asarray(lat), np.asarray(lon), con_distanze=False)[FORMA_RILIEVO]
    return pd.concat([f, forma.set_axis(f.index)], axis=1)


def correggi(modello, f, param, doy):
    """Applica il modello alle feature f (DataFrame, una riga per punto); restituisce il valore corretto."""
    f = f.assign(aurora=f[f"a_{param}"], doy_sin=np.sin(2 * np.pi * doy / 365.25),
                 doy_cos=np.cos(2 * np.pi * doy / 365.25))
    return prevedi(modello, f, param)


def griglia_dem():
    """Griglia ~RIS gradi sull'estensione del DEM, quota = DEM a 30 m nel centro del pixel.
    Nel punto, non media del pixel: come per le stazioni (dem_elevation) e come la forma del rilievo,
    cosi' ogni pixel e' il valore atteso in un punto al centro della cella, con feature dello stesso
    tipo di quelle del training."""
    with rasterio.open(DEM) as d:
        b = d.bounds
    ny, nx = int(round((b.top - b.bottom) / RIS)), int(round((b.right - b.left) / RIS))
    lat = b.top - (np.arange(ny) + 0.5) * (b.top - b.bottom) / ny
    lon = b.left + (np.arange(nx) + 0.5) * (b.right - b.left) / nx
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    elev = dem_elevation(LAT.ravel(), LON.ravel()).reshape(ny, nx)
    return lat, lon, elev


def osservazioni(giorni):
    """Osservazioni dei giorni richiesti, con le stesse regole del notebook §6.6."""
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
    # ARPAE (radiazione Emilia-Romagna, notebook §6.3): file mensili gia' giornalieri; l'ultimo giorno
    # di un mese compare anche nel file del mese dopo, quindi si ricombinano le ore prima del filtro
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
    print("fattori di scala:", FATTORI)
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
            print(f"salto {day.date()}: manca {_file_campi(day.date())}")
            continue
        campi = campi.sel(date=[day])
        mancanti = [c for c in teste["tx3"]["feature_cols"]
                    if c.startswith(("sup_", "atm_")) and c.rsplit("_", 1)[0] + "_media" not in campi
                    and c not in campi]
        if mancanti:
            print(f"salto {day.date()}: il file dei campi non ha tutte le variabili ({mancanti[:3]}...): "
                  f"va riestratto con la versione attuale di extract_aurora_campi.py")
            continue
        doy = day.dayofyear
        f_griglia = feature_punti(campi, LAT.ravel(), LON.ravel(), gelev.ravel())
        out = xr.Dataset(coords={"lat": glat, "lon": glon})
        out["elev"] = (("lat", "lon"), gelev)
        fig, axes = plt.subplots(len(PARAMS), 3, figsize=(17, 4.2 * len(PARAMS)))
        for r, p in enumerate(PARAMS):
            modello = teste[p]
            campo = (campi[COL_AURORA[p]].isel(date=0) * FATTORI[p]).clip(*TABLE_QC[p])
            a25 = campo.interp(lat=glat, lon=glon, method="nearest").values           # Aurora 0.25 gradi
            a1 = f_griglia[f"a_{p}"].values.reshape(LAT.shape)                         # Aurora interpolata
            c1 = correggi(modello, f_griglia, p, doy).reshape(LAT.shape)               # downscaling
            out[f"{p}_aurora_25km"] = (("lat", "lon"), a25)
            out[f"{p}_aurora_bilineare"] = (("lat", "lon"), a1)
            out[f"{p}_downscaling"] = (("lat", "lon"), c1)

            # confronto con le stazioni di quel giorno
            o = obs[(obs["parameter"] == p) & (obs["date"] == day)].copy()
            if len(o):
                el = dem_elevation(o["lat"].values, o["lon"].values)
                o = o[~np.isnan(el)]
                el = el[~np.isnan(el)]
                fo = feature_punti(campi, o["lat"].values, o["lon"].values, el)
                ok = fo[[c for c in modello["feature_cols"] if c not in ("aurora", "doy_sin", "doy_cos")]
                        ].notna().all(axis=1).values
                o, fo = o[ok], fo[ok]
                for nome, v in {"Aurora (bilineare)": fo[f"a_{p}"].values,
                                "downscaling": correggi(modello, fo, p, doy)}.items():
                    e = v - o["obs"].values
                    confronto.append({"giorno": str(day.date()), "parametro": p, "versione": nome, "n": len(e),
                                      "rmse": float(np.sqrt(np.mean(e ** 2))), "bias": float(e.mean())})

            # figura: Aurora 25 km | Aurora interpolata | downscaling ~1 km, stazioni sovrapposte
            vmin, vmax = np.nanpercentile(np.concatenate([a25.ravel(), c1.ravel()]), [2, 98])
            cmap = {"tx3": "RdYlBu_r", "tn3": "RdYlBu_r", "qq2": "magma", "rr7": "Blues"}[p]
            ext = [glon.min(), glon.max(), glat.min(), glat.max()]
            for c, (titolo, m) in enumerate([("Aurora 0.25°", a25), ("Aurora interpolata (bilineare)", a1),
                                              ("Downscaling ~1 km", c1)]):
                ax = axes[r, c]
                im = ax.imshow(m, extent=ext, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax,
                               aspect=1 / np.cos(np.radians(45)), interpolation="nearest")
                regioni.boundary.plot(ax=ax, color="k", linewidth=0.4, aspect=None)
                if c == 0:
                    # punti della griglia di Aurora: ogni cella e' l'area piu' vicina a un punto
                    plon, plat = np.meshgrid(campo["lon"].values, campo["lat"].values)
                    ax.scatter(plon, plat, s=3, color="k", linewidth=0)
                if c == 2 and len(o):
                    ax.scatter(o["lon"], o["lat"], c=o["obs"], cmap=cmap, vmin=vmin, vmax=vmax, s=12,
                               edgecolor="k", linewidth=0.3)
                ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
                ax.set_title(f"{p} - {titolo}", fontsize=9)
            fig.colorbar(im, ax=axes[r, :], shrink=0.8, label=f"{p} ({UNITA[p]})")
        fig.suptitle(f"{day.date()}: Aurora 25 km e downscaling ~1 km (a sinistra: punti della griglia di "
                     f"Aurora; a destra: stazioni osservate)", fontsize=13)
        fig.savefig(f"mappe/mappa_{day.date()}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        out.to_netcdf(f"mappe/mappa_{day.date()}.nc")
        print(f"salvati mappa_{day.date()}.png e mappa_{day.date()}.nc")

    if confronto:
        c = pd.DataFrame(confronto)
        c.to_csv("risultati/mappa_confronto_stazioni.csv", index=False)
        print("\nconfronto con le stazioni (RMSE):")
        print(c.pivot_table(index=["giorno", "parametro"], columns="versione", values="rmse").round(2))


if __name__ == "__main__":
    main()
