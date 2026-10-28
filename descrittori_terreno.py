"""
Terrain descriptors for the downscaling head, in addition to elevation (point 2 of the experiments).
The same function is used for the stations (training/test) and for the map pixels, so the features
are computed in the same way.

From the Copernicus GLO-30 DEM (cop30_padania.tif), resampled to ~90 m (mean of 3 x 3 pixels):
  tpi_1km, tpi_5km, tpi_20km  position in the terrain: elevation of the point minus mean elevation in a
                              square of side ~2, ~10, ~40 km around it. Negative = valley floor or basin
                              (cold air at night, inversions), positive = ridge or upper slope.
  pendenza                    slope, degrees, at ~90 m
  nordness, eastness          cos/sin of the aspect x sin(slope): north/south, east/west facing slopes
                              (0 on flat ground)
From Natural Earth 10m:
  dist_mare_km                distance from the coastline
  dist_lago_km                distance from the nearest lake (large lakes: Garda, Maggiore, Como, ...)

The final model uses only the 6 descriptors of FORMA_RILIEVO (terrain shape), and only for tn3 and
qq2 (esperimento_terreno.py: tn3 -2.6%, qq2 -0.8%; no gain for tx3 and rr7; the distances from sea
and lakes make the temperatures worse). The notebook (§6.6) adds them to the training/test table
with descrittori(); mappa_downscaling.py computes them in the same way in each pixel.

Usage:
    python descrittori_terreno.py      # computes and saves the station descriptors (for the experiments)
Output:
    descrittori_terreno_stazioni.parquet (uid + descriptors)
"""
from functools import lru_cache

import numpy as np
import pandas as pd

# Paths relative to the main project folder (where the script is run from)
DEM = "dati/statici/cop30_padania.tif"
TABELLA_TRAINING = "tabelle/training_table_aurora.parquet"

FATTORE = 3                     # 30 m -> ~90 m
LATI_TPI_KM = {"tpi_1km": 2, "tpi_5km": 10, "tpi_20km": 40}
DESCRITTORI = list(LATI_TPI_KM) + ["pendenza", "nordness", "eastness", "dist_mare_km", "dist_lago_km"]
FORMA_RILIEVO = ["tpi_1km", "tpi_5km", "tpi_20km", "pendenza", "nordness", "eastness"]
USCITA = "tabelle/descrittori_terreno_stazioni.parquet"


def dem_elevation(lats, lons):
    """Elevation (m) from the 30 m Copernicus GLO-30 DEM, point by point (feature "elev" of stations
    and map pixels). NaN outside the DEM (bbox 44.0-46.5 N, 6.5-13.0 E). Against the official
    elevation of the ECA&D stations: median error ~2 m (the old elev_cerra, 5.5 km CERRA cell, was
    ~50 m, with peaks above 800 m in the mountains)."""
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
    """Elevation at ~90 m and the derived descriptors, as grids over the DEM area."""
    import rasterio
    from rasterio.enums import Resampling
    from scipy.ndimage import uniform_filter

    with rasterio.open(DEM) as d:
        h, w = d.height // FATTORE, d.width // FATTORE
        z = d.read(1, out_shape=(h, w), resampling=Resampling.average).astype("float64")
        b = d.bounds
    lat_c = (b.top + b.bottom) / 2
    dy = (b.top - b.bottom) / h * 111_320                         # metres per pixel in latitude
    dx = (b.right - b.left) / w * 111_320 * np.cos(np.radians(lat_c))
    g = {}
    for nome, lato_km in LATI_TPI_KM.items():
        finestra = (max(1, round(lato_km * 1000 / dy)), max(1, round(lato_km * 1000 / dx)))
        g[nome] = z - uniform_filter(z, size=finestra, mode="nearest")
    gy, gx = np.gradient(z, dy, dx)                               # rows: from north to south
    pend = np.arctan(np.hypot(gx, gy))
    # aspect: direction the slope faces downhill (0 = north, 90 = east)
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
    """Terrain descriptors at the given points (degrees). NaN outside the DEM.
    con_distanze=False skips the distances from sea and lakes (not used by the final model)."""
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
    print(f"saved {USCITA}: {len(d)} stations")
    print(d[DESCRITTORI].describe().round(2).T)


if __name__ == "__main__":
    main()
