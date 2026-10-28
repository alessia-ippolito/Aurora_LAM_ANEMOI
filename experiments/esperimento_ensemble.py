"""
Experiment (point 1): mean of several networks trained with different seeds (ensemble), on the final
configuration chosen with the previous experiments:
  - features: the current 28 + terrain shape for tn3 and qq2 (esperimento_terreno.py)
  - precipitation: plain residual with squared error (esperimento_pioggia.py)
  - 64-64 network with dropout 0.1 (esperimento_iperparametri.py)
Region x year CV of cv_comune.py. For each fold N_SEED networks are trained; the ensemble is the mean
of their predictions. The epoch is chosen on the mean over the folds, as always.

Usage:
    python esperimento_ensemble.py
Output:
    esperimento_ensemble_risultati.csv  (per fold and epoch: single network, mean over seeds, ensemble of k networks)
"""
import numpy as np
import pandas as pd
import torch

from cv_comune import PARAMS, allena_valuta, carica_tabella, errori, folds
from descrittori_terreno import USCITA
from experiments.esperimento_terreno import FORMA
from train_downscaling_head import FEATURE_COLS

N_SEED = 5
FEATURE = {p: FEATURE_COLS + (FORMA if p in ("tn3", "qq2") else []) for p in PARAMS}
MAX_EPOCHE = {"tx3": 20, "tn3": 20, "rr7": 20, "qq2": 45}
OPZIONI = dict(hidden=(64, 64), dropout=0.1)


def main():
    torch.set_num_threads(6)
    t = carica_tabella(colonne_extra=pd.read_parquet(USCITA))
    righe = []
    for param in PARAMS:
        for reg, anno, tr, va in folds(t[t["parameter"] == param]):
            ov = va["obs"].values
            pred = np.stack([allena_valuta(tr, va, FEATURE[param], param, max_epoche=MAX_EPOCHE[param],
                                           seed=s, salva_pred=True, **OPZIONI)[1] for s in range(N_SEED)])
            # pred: seed x epoch x rows
            for ep in range(pred.shape[1]):
                singole = [errori(pred[s, ep], ov)["rmse"] for s in range(N_SEED)]
                riga = {"parametro": param, "regione": reg, "anno": anno, "epoca": ep + 1,
                        "rmse_singola_media": float(np.mean(singole))}
                for k in range(2, N_SEED + 1):
                    riga[f"rmse_ensemble_{k}"] = errori(pred[:k, ep].mean(axis=0), ov)["rmse"]
                riga["bias_ensemble"] = errori(pred[:, ep].mean(axis=0), ov)["bias"]
                righe.append(riga)
            print(f"[{param}] {reg} {anno} done", flush=True)
            pd.DataFrame(righe).to_csv("risultati/esperimento_ensemble_risultati.csv", index=False)

    res = pd.DataFrame(righe)
    curva = res.groupby(["parametro", "epoca"]).mean(numeric_only=True).drop(columns="anno")
    out = {}
    for param in PARAMS:
        c = curva.loc[param]
        out[param] = {"single network": c["rmse_singola_media"].min(),
                      "single epoch": int(c["rmse_singola_media"].idxmin()),
                      **{f"ensemble {k}": c[f"rmse_ensemble_{k}"].min() for k in range(2, N_SEED + 1)},
                      f"ensemble epoch {N_SEED}": int(c[f"rmse_ensemble_{N_SEED}"].idxmin())}
    tab = pd.DataFrame(out)
    print("\n=== Region x year CV: mean RMSE over the 9 folds at the best epoch ===")
    print(tab.round(3).to_string())
    g = 100 * (tab.loc[f"ensemble {N_SEED}"] - tab.loc["single network"]) / tab.loc["single network"]
    print(f"\ngain of the ensemble of {N_SEED} networks over the single network (%):", g.round(2).to_dict())


if __name__ == "__main__":
    main()
