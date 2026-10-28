"""
Experiment (point 2): does the network improve with terrain descriptors in addition to elevation?
(descrittori_terreno.py; region x year CV of cv_comune.py; 2 seeds per variant)

  C        current network (28 features)
  T1       C + terrain shape: tpi_1km, tpi_5km, tpi_20km, pendenza (slope), nordness, eastness
  T2       T1 + distance from sea and lakes
           (fields that are smooth in space: they could let the network recognise the station
           location as lat/lon did; this is why they are a separate variant)

Usage:
    python esperimento_terreno.py
Output:
    esperimento_terreno_risultati.csv
"""
import pandas as pd
import torch

from cv_comune import PARAMS, carica_tabella, cv_rete, riepilogo
from descrittori_terreno import USCITA
from train_downscaling_head import FEATURE_COLS

SEEDS = [0, 1]
FORMA = ["tpi_1km", "tpi_5km", "tpi_20km", "pendenza", "nordness", "eastness"]
ACQUA = ["dist_mare_km", "dist_lago_km"]
VARIANTI = {"C": FEATURE_COLS, "T1 forma del rilievo": FEATURE_COLS + FORMA,
            "T2 forma + acqua": FEATURE_COLS + FORMA + ACQUA}


def main():
    torch.set_num_threads(6)
    t = carica_tabella(colonne_extra=pd.read_parquet(USCITA))
    righe = []
    for param in PARAMS:
        for nome, feature in VARIANTI.items():
            for seed in SEEDS:
                for r in cv_rete(t, param, feature, nome, seed=seed):
                    righe.append({**r, "seed": seed})
                print(f"[{param}] {nome} seed {seed} done", flush=True)
                pd.DataFrame(righe).to_csv("risultati/esperimento_terreno_risultati.csv", index=False)

    res = pd.DataFrame(righe)
    s = riepilogo(res)
    print("\n=== Region x year CV: mean RMSE over folds and seeds, at the best epoch ===")
    print(s["rmse"].unstack("parametro")[list(PARAMS)].round(3))
    print("\nbest epoch:")
    print(s["epoca"].unstack("parametro")[list(PARAMS)])
    # per fold, at the respective best epoch: in how many folds T1/T2 beat C
    sel = res.merge(s["epoca"].rename("ep").reset_index(), on=["variante", "parametro"])
    sel = sel[sel["epoca"] == sel["ep"]].groupby(["parametro", "regione", "anno", "variante"])["rmse"].mean()
    tab = sel.unstack("variante")
    for v in list(VARIANTI)[1:]:
        print(f"folds where {v} beats C:", (tab[v] < tab["C"]).groupby("parametro").sum().to_dict())
    print("\nby fold:")
    print(tab.round(3))


if __name__ == "__main__":
    main()
