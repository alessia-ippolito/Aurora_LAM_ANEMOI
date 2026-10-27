"""
Descrittori del terreno per la testa di downscaling, oltre alla quota (punto 2 degli esperimenti).
Stessa funzione per le stazioni (training/test) e per i pixel delle mappe, cosi' le feature sono
calcolate nello stesso modo.

Dal Copernicus DEM GLO-30 (cop30_padania.tif), ricampionato a ~90 m (media di 3 x 3 pixel):
  tpi_1km, tpi_5km, tpi_20km  posizione nel rilievo: quota del punto meno quota media in un quadrato
                              di lato ~2, ~10, ~40 km intorno. Negativo = fondovalle o conca (aria
                              fredda di notte, inversioni), positivo = dorsale o versante alto.
  pendenza                    gradi, a ~90 m
  nordness, eastness          cos/sin dell'esposizione x sin(pendenza): versanti a nord/sud, est/ovest
                              (0 in piano)
Da Natural Earth 10m:
  dist_mare_km                distanza dalla linea di costa
  dist_lago_km                distanza dal lago piu' vicino (grandi laghi: Garda, Maggiore, Como, ...)

Nel modello finale entrano solo i 6 descrittori di FORMA_RILIEVO, e solo per tn3 e qq2
(esperimento_terreno.py: tn3 -2.6%, qq2 -0.8%; nessun guadagno per tx3 e rr7; le distanze da mare e
laghi peggiorano le temperature). Il notebook (§6.6) li aggiunge alla tabella di training/test con
descrittori(); mappa_downscaling.py li calcola nello stesso modo in ogni pixel.

Uso:
    python descrittori_terreno.py      # calcola e salva i descrittori delle stazioni (per gli esperimenti)
Output:
    descrittori_terreno_stazioni.parquet (uid + descrittori)
"""
from functools import lru_cache

import numpy as np
import pandas as pd

# Percorsi relativi alla cartella principale del progetto (da cui si lancia lo script)
DEM = "dati/statici/cop30_padania.tif"
TABELLA_TRAINING = "tabelle/training_table_aurora.parquet"

FATTORE = 3                     # 30 m -> ~90 m
LATI_TPI_KM = {"tpi_1km": 2, "tpi_5km": 10, "tpi_20km": 40}
DESCRITTORI = list(LATI_TPI_KM) + ["pendenza", "nordness", "eastness", "dist_mare_km", "dist_lago_km"]
FORMA_RILIEVO = ["tpi_1km", "tpi_5km", "tpi_20km", "pendenza", "nordness", "eastness"]
USCITA = "tabelle/descrittori_terreno_stazioni.parquet"


def dem_elevation(lats, lons):
    """Quota (m) dal Copernicus DEM GLO-30 a 30 m, punto per punto (feature "elev" di stazioni e
    pixel delle mappe). NaN se fuori dal DEM (bbox 44.0-46.5 N, 6.5-13.0 E). Rispetto alla quota
    ufficiale delle stazioni ECA&D: errore mediano ~2 m (la vecchia elev_cerra, cella CERRA 5.5 km,
    era ~50 m, con punte oltre 800 m in montagna)."""
    import rasterio

    with rasterio.open(DEM) as d:
        b = d.bounds
        out = []
        for lat, lon, v in zip(lats, lons, d.sample(list(zip(lons, lats)))):
            inside = b.left < lon < b.right and b.bottom < lat < b.top
            out.append(float(v[0]) if inside else np.nan)
    return np.array(out)


@lru_cache(maxsize=1)
def _griglie_dem():
    """Quota a ~90 m e i descrittori derivati, come griglie sull'area del DEM."""
    import rasterio
    from rasterio.enums import Resampling
    from scipy.ndimage import uniform_filter

    with rasterio.open(DEM) as d:
        h, w = d.height // FATTORE, d.width // FATTORE
        z = d.read(1, out_shape=(h, w), resampling=Resampling.average).astype("float64")
        b = d.bounds
    lat_c = (b.top + b.bottom) / 2
    dy = (b.top - b.bottom) / h * 111_320                         # metri per pixel in latitudine
    dx = (b.right - b.left) / w * 111_320 * np.cos(np.radians(lat_c))
    g = {}
    for nome, lato_km in LATI_TPI_KM.items():
        finestra = (max(1, round(lato_km * 1000 / dy)), max(1, round(lato_km * 1000 / dx)))
        g[nome] = z - uniform_filter(z, size=finestra, mode="nearest")
    gy, gx = np.gradient(z, dy, dx)                               # righe: da nord a sud
    pend = np.arctan(np.hypot(gx, gy))
    # esposizione: direzione verso cui scende il versante (0 = nord, 90 = est)
    esp = np.arctan2(-gx, gy)
    g["pendenza"] = np.degrees(pend)
    g["nordness"] = np.cos(esp) * np.sin(pend)
    g["eastness"] = np.sin(esp) * np.sin(pend)
    return g, b, h, w


def _distanze(lats, lons):
    import cartopy.io.shapereader as shpreader
    import geopandas as gpd
    import shapely
    from shapely.geometry import box

    area = box(4.5, 42.0, 15.0, 48.0)
    punti = gpd.GeoSeries(gpd.points_from_xy(lons, lats), crs="EPSG:4326").to_crs(32632)
    out = {}
    for nome, cat, layer in [("dist_mare_km", "physical", "coastline"), ("dist_lago_km", "physical", "lakes")]:
        geo = gpd.read_file(shpreader.natural_earth("10m", cat, layer))
        geo = geo[geo.intersects(area)].to_crs(32632)
        unione = shapely.union_all(geo.geometry.values)
        if layer == "lakes":
            unione = unione.boundary
        out[nome] = shapely.distance(punti.values, unione) / 1000
    return out


def descrittori(lats, lons, con_distanze=True):
    """Descrittori del terreno nei punti dati (gradi). NaN fuori dal DEM.
    con_distanze=False salta le distanze da mare e laghi (non usate dal modello finale)."""
    lats, lons = np.asarray(lats, float), np.asarray(lons, float)
    g, b, h, w = _griglie_dem()
    r = np.floor((b.top - lats) / (b.top - b.bottom) * h).astype(int)
    c = np.floor((lons - b.left) / (b.right - b.left) * w).astype(int)
    dentro = (r >= 0) & (r < h) & (c >= 0) & (c < w)
    rr, cc = np.clip(r, 0, h - 1), np.clip(c, 0, w - 1)
    out = {k: np.where(dentro, v[rr, cc], np.nan) for k, v in g.items()}
    if not con_distanze:
        return pd.DataFrame(out)[[c for c in DESCRITTORI if c in out]]
    out.update(_distanze(lats, lons))
    return pd.DataFrame(out)[DESCRITTORI]


def main():
    t = pd.read_parquet(TABELLA_TRAINING, columns=["uid", "lat", "lon"])
    st = t.drop_duplicates("uid").reset_index(drop=True)
    d = descrittori(st["lat"].values, st["lon"].values)
    d.insert(0, "uid", st["uid"].values)
    d.to_parquet(USCITA, index=False)
    print(f"salvato {USCITA}: {len(d)} stazioni")
    print(d[DESCRITTORI].describe().round(2).T)


if __name__ == "__main__":
    main()
