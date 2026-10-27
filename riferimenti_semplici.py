"""
Riferimenti semplici per capire quanto vale la rete (CV regione x anno, cv_comune.py):

  R0  Aurora interpolata              nessuna correzione
  R1  + correzione di quota           tx3/tn3: Aurora - 6.5 gradi/km x dz (dz = quota stazione -
                                      quota del terreno di Aurora); rr7/qq2 come R0
  R2  + correzione media mensile      R1 + errore medio per mese, stimato sui dati di training del fold
  R3  regressione lineare             correzione = combinazione lineare delle stesse 28 feature
                                      della rete, stimata sul training del fold
  C   rete attuale                    28 feature, 64-64 (ricalcolata con cv_comune.py: e' il
                                      riferimento per gli esperimenti successivi)

Per rr7 e qq2 i valori negativi sono portati a 0, come nella rete.

Uso:
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

GRADIENTE = -0.0065   # gradi per metro


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
        print(f"[{param}] riferimenti fatti; ora la rete attuale...", flush=True)
        righe += cv_rete(t, param, FEATURE_COLS, "C rete attuale")
        pd.DataFrame(righe).to_csv("riferimenti_semplici_risultati.csv", index=False)

    res = pd.DataFrame(righe)
    s = riepilogo(res)
    print("\n=== CV regione x anno: RMSE medio sui 9 fold ===")
    print(s["rmse"].unstack("parametro")[list(PARAMS)].round(3))
    print("\nbias medio:")
    print(s["bias"].unstack("parametro")[list(PARAMS)].round(3))
    print("\nepoca scelta per la rete:", s.xs("C rete attuale")["epoca"].to_dict())


if __name__ == "__main__":
    main()
