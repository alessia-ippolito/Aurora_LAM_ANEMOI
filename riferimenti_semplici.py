"""
Simple baselines to judge how much the network is worth (region x year CV, cv_comune.py):

  R0  Aurora interpolated             no correction
  R1  + elevation correction          tx3/tn3: Aurora - 6.5 degrees/km x dz (dz = station elevation -
                                      Aurora terrain elevation); rr7/qq2 as R0
  R2  + monthly mean correction       R1 + mean error per month, estimated on the training data of the fold
  R3  linear regression               correction = linear combination of the same 28 features as the
                                      network, estimated on the training data of the fold
  C   current network                 28 features, 64-64 (recomputed with cv_comune.py: it is the
                                      reference for the later experiments)

For rr7 and qq2 negative values are set to 0, as in the network.
The variant labels (e.g. "R0 Aurora interpolata") are kept in Italian: they are the keys stored in
the results file.

Usage:
    python riferimenti_semplici.py
Output:
    riferimenti_semplici_risultati.csv
"""
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression

from cv_comune import NON_NEGATIVI, PARAMS, carica_tabella, cv_rete, errori, folds, riepilogo
from train_downscaling_head import FEATURE_COLS

GRADIENTE = -0.0065   # degrees per metre


def main():
    torch.set_num_threads(6)
    t = carica_tabella()
    righe = []
    for param in PARAMS:
        tp = t[t["parameter"] == param]
        for reg, anno, tr, va in folds(tp):
            def aggiungi(variante, pred):
                if param in NON_NEGATIVI:
                    pred = np.clip(pred, 0, None)
                righe.append({"variante": variante, "parametro": param, "regione": reg, "anno": anno,
                              "epoca": 0, **errori(pred, va["obs"].values)})
            quota = param in ("tx3", "tn3")
            r1_tr = tr["aurora"] + (GRADIENTE * tr["dz"] if quota else 0)
            r1_va = va["aurora"] + (GRADIENTE * va["dz"] if quota else 0)
            aggiungi("R0 Aurora interpolata", va["aurora"].values)
            aggiungi("R1 + correzione di quota", r1_va.values)
            mensile = (tr["obs"] - r1_tr).groupby(tr["mese"]).mean()
            aggiungi("R2 + correzione media mensile", (r1_va + va["mese"].map(mensile)).values)
            lin = LinearRegression().fit(tr[FEATURE_COLS].values, tr["residual"].values)
            aggiungi("R3 regressione lineare", va["aurora"].values + lin.predict(va[FEATURE_COLS].values))
        print(f"[{param}] baselines done; now the current network...", flush=True)
        righe += cv_rete(t, param, FEATURE_COLS, "C rete attuale")
        pd.DataFrame(righe).to_csv("risultati/riferimenti_semplici_risultati.csv", index=False)

    res = pd.DataFrame(righe)
    s = riepilogo(res)
    print("\n=== Region x year CV: mean RMSE over the 9 folds ===")
    print(s["rmse"].unstack("parametro")[list(PARAMS)].round(3))
    print("\nmean bias:")
    print(s["bias"].unstack("parametro")[list(PARAMS)].round(3))
    print("\nepoch chosen for the network:", s.xs("C rete attuale")["epoca"].to_dict())


if __name__ == "__main__":
    main()
