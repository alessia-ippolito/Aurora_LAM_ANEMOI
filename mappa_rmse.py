"""
Map of the per-station error on the test set (Piedmont 2020): RMSE of interpolated Aurora, RMSE of the
final model and their difference, one point per station. The RMSE exists only where there are
observations, so the map is by station and not by pixel.

Uses the networks already saved (checkpoints/<p>_finale.pt) on the same rows as the test evaluation
(train_downscaling_head.py --valuta_test, exclusions included): it trains and chooses nothing.
Stations with fewer than MIN_GIORNI days are not shown (unreliable RMSE).

Usage:
    python mappa_rmse.py
Output:
    mappa_rmse_test.png, rmse_stazioni_test.csv
"""
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from train_downscaling_head import PARAMS, carica_modello, load_tables, prevedi

# Paths relative to the main project folder (where the script is run from)
REGIONI = "dati/statici/italy_regions.geojson"

MIN_GIORNI = 30
UNITA = {"tx3": "°C", "tn3": "°C", "qq2": "W/m²", "rr7": "mm"}


def rmse_stazioni(param):
    df = load_tables(param)
    df = df[df["split"] == "test"].copy()
    df["modello"] = prevedi(carica_modello(param), df, param)
    df["e_aurora"] = (df["aurora"] - df["obs"]) ** 2
    df["e_modello"] = (df["modello"] - df["obs"]) ** 2
    s = df.groupby("uid").agg(lat=("lat", "first"), lon=("lon", "first"), elev=("elev", "first"),
                              giorni=("obs", "size"), rmse_aurora=("e_aurora", "mean"),
                              rmse_modello=("e_modello", "mean"))
    s[["rmse_aurora", "rmse_modello"]] = np.sqrt(s[["rmse_aurora", "rmse_modello"]])
    s["differenza"] = s["rmse_modello"] - s["rmse_aurora"]
    return s[s["giorni"] >= MIN_GIORNI].reset_index().assign(parametro=param)


def main():
    tab = pd.concat([rmse_stazioni(p) for p in PARAMS], ignore_index=True)
    tab.to_csv("risultati/rmse_stazioni_test.csv", index=False)
    regioni = gpd.read_file(REGIONI)

    fig, axes = plt.subplots(len(PARAMS), 3, figsize=(13, 4 * len(PARAMS)))
    for riga, p in zip(axes, PARAMS):
        s = tab[tab["parametro"] == p]
        vmax = np.nanpercentile(s[["rmse_aurora", "rmse_modello"]].values, 98)
        dmax = np.nanpercentile(s["differenza"].abs(), 98)
        for ax, col, cmap, vmin, vm, titolo in [
                (riga[0], "rmse_aurora", "viridis", 0, vmax, "RMSE interpolated Aurora"),
                (riga[1], "rmse_modello", "viridis", 0, vmax, "RMSE model"),
                (riga[2], "differenza", "RdBu_r", -dmax, dmax, "model - Aurora (blue = better)")]:
            regioni.boundary.plot(ax=ax, color="#888888", linewidth=0.4, aspect=None)
            sc = ax.scatter(s["lon"], s["lat"], c=s[col], cmap=cmap, vmin=vmin, vmax=vm, s=22,
                            edgecolor="k", linewidth=0.3, zorder=3)
            ax.set_xlim(6.6, 9.3); ax.set_ylim(44.0, 46.5)
            ax.set_aspect(1 / np.cos(np.radians(45)))
            mediana = s[col].median()
            ax.set_title(f"{p} - {titolo}\nmedian per station {mediana:.2f} {UNITA[p]}", fontsize=9)
            fig.colorbar(sc, ax=ax, shrink=0.8, label=UNITA[p])
    fig.suptitle(f"Test Piedmont 2020: error per station (stations with at least {MIN_GIORNI} days)")
    fig.tight_layout()
    fig.savefig("mappe/mappa_rmse_test.png", dpi=130)
    print("saved mappa_rmse_test.png and rmse_stazioni_test.csv")
    print(tab.groupby("parametro")[["rmse_aurora", "rmse_modello", "differenza"]].median().round(3))
    print("stations where the model improves on Aurora:",
          tab.assign(m=tab["differenza"] < 0).groupby("parametro")["m"].agg(["sum", "size"]).to_dict("index"))


if __name__ == "__main__":
    main()
