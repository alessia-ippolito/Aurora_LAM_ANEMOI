"""
Esperimento (punto 2): la rete migliora con descrittori del terreno oltre alla quota?
(descrittori_terreno.py; CV regione x anno di cv_comune.py; 2 seed per variante)

  C        rete attuale (28 feature)
  T1       C + forma del rilievo: tpi_1km, tpi_5km, tpi_20km, pendenza, nordness, eastness
  T2       T1 + distanza da mare e laghi
           (campi lisci nello spazio: potrebbero far riconoscere la posizione delle stazioni come
           facevano lat/lon; per questo sono una variante a parte)

Uso:
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
                print(f"[{param}] {nome} seed {seed} fatto", flush=True)
                pd.DataFrame(righe).to_csv("esperimento_terreno_risultati.csv", index=False)

    res = pd.DataFrame(righe)
    s = riepilogo(res)
    print("\n=== CV regione x anno: RMSE medio sui fold e sui seed, all'epoca migliore ===")
    print(s["rmse"].unstack("parametro")[list(PARAMS)].round(3))
    print("\nepoca migliore:")
    print(s["epoca"].unstack("parametro")[list(PARAMS)])
    # per fold, alla rispettiva epoca migliore: in quanti fold T1/T2 battono C
    sel = res.merge(s["epoca"].rename("ep").reset_index(), on=["variante", "parametro"])
    sel = sel[sel["epoca"] == sel["ep"]].groupby(["parametro", "regione", "anno", "variante"])["rmse"].mean()
    tab = sel.unstack("variante")
    for v in list(VARIANTI)[1:]:
        print(f"fold in cui {v} batte C:", (tab[v] < tab["C"]).groupby("parametro").sum().to_dict())
    print("\nper fold:")
    print(tab.round(3))


if __name__ == "__main__":
    main()
