"""
Esperimento (punto 1): media di piu' reti allenate con seed diversi (ensemble), sulla configurazione
finale scelta con gli esperimenti precedenti:
  - feature: le 28 attuali + forma del rilievo per tn3 e qq2 (esperimento_terreno.py)
  - pioggia: residuo semplice con errore quadratico (esperimento_pioggia.py)
  - rete 64-64 con dropout 0.1 (esperimento_iperparametri.py)
CV regione x anno di cv_comune.py. Per ogni fold si allenano N_SEED reti; l'ensemble e' la media
delle loro previsioni. L'epoca si sceglie in media sui fold, come sempre.

Uso:
    python esperimento_ensemble.py
Output:
    esperimento_ensemble_risultati.csv  (per fold ed epoca: rete singola, media sui seed, ensemble di k reti)
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
            # pred: seed x epoca x righe
            for ep in range(pred.shape[1]):
                singole = [errori(pred[s, ep], ov)["rmse"] for s in range(N_SEED)]
                riga = {"parametro": param, "regione": reg, "anno": anno, "epoca": ep + 1,
                        "rmse_singola_media": float(np.mean(singole))}
                for k in range(2, N_SEED + 1):
                    riga[f"rmse_ensemble_{k}"] = errori(pred[:k, ep].mean(axis=0), ov)["rmse"]
                riga["bias_ensemble"] = errori(pred[:, ep].mean(axis=0), ov)["bias"]
                righe.append(riga)
            print(f"[{param}] {reg} {anno} fatto", flush=True)
            pd.DataFrame(righe).to_csv("risultati/esperimento_ensemble_risultati.csv", index=False)

    res = pd.DataFrame(righe)
    curva = res.groupby(["parametro", "epoca"]).mean(numeric_only=True).drop(columns="anno")
    out = {}
    for param in PARAMS:
        c = curva.loc[param]
        out[param] = {"rete singola": c["rmse_singola_media"].min(),
                      "epoca singola": int(c["rmse_singola_media"].idxmin()),
                      **{f"ensemble {k}": c[f"rmse_ensemble_{k}"].min() for k in range(2, N_SEED + 1)},
                      f"epoca ensemble {N_SEED}": int(c[f"rmse_ensemble_{N_SEED}"].idxmin())}
    tab = pd.DataFrame(out)
    print("\n=== CV regione x anno: RMSE medio sui 9 fold all'epoca migliore ===")
    print(tab.round(3).to_string())
    g = 100 * (tab.loc[f"ensemble {N_SEED}"] - tab.loc["rete singola"]) / tab.loc["rete singola"]
    print(f"\nguadagno dell'ensemble di {N_SEED} reti rispetto alla rete singola (%):", g.round(2).to_dict())


if __name__ == "__main__":
    main()
