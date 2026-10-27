"""
Importanza delle feature (permutation importance) del modello finale, per singola feature e per gruppo.

In ogni fold della CV regione x anno (cv_comune.py) si allena una rete con la configurazione finale
(train_downscaling_head.py: feature per variabile, 64-64, dropout 0.1, epoche da epoche_cv.json).
Poi, sui dati di validazione del fold, si mescolano a caso i valori di una feature (o di tutte le
feature di un gruppo, con la stessa permutazione delle righe) e si misura di quanto sale l'RMSE.
Importanza = aumento percentuale dell'RMSE, media su RIPETIZIONI permutazioni e sui 9 fold.

Per gruppo e per singola feature, perche' molte feature sono quasi copie l'una dell'altra (es.
aurora = a_<variabile>, temperatura a 925 e 850 hPa con r = 0.98): mescolandone una sola il modello
usa ancora la gemella e l'importanza risulta bassa anche se l'informazione e' fondamentale; mescolando
tutto il gruppo si misura quanto conta l'informazione.

Attenzione: la permutazione crea combinazioni irrealistiche (es. 30 gradi in superficie e 0 a 850
hPa). L'importanza dice quanto il modello USA un'informazione, non che quella informazione sia la
causa fisica dell'errore di Aurora.

Uso:
    python esperimento_importanza.py
Output:
    esperimento_importanza_risultati.csv
"""
import json

import numpy as np
import pandas as pd
import torch

from cv_comune import NON_NEGATIVI, PARAMS, carica_tabella, folds
from train_downscaling_head import FORMA_RILIEVO, allena_rete, feature_cols

RIPETIZIONI = 3


def gruppi(param):
    principali = [f"a_{p}" for p in PARAMS]
    g = {
        "Aurora: variabile stessa": [c for c in ("aurora", f"a_{param}") if c in feature_cols(param)],
        "Aurora: altre 3 principali": [c for c in principali if c != f"a_{param}"],
        "Temperatura in superficie": ["sup_2t_media"],
        "Profilo termico in quota": ["atm_t_925", "atm_t_850", "atm_t_700", "atm_t_500"],
        "Umidita'": ["atm_q_925", "atm_q_850", "atm_q_700"],
        "Vento": ["sup_10u_media", "sup_10v_media", "sup_vento10_max", "sup_i10fg_media", "atm_u_850", "atm_v_850"],
        "Pressione e geopotenziale": ["sup_msl_media", "atm_z_500"],
        "Nubi (infrarosso uscente)": ["sup_ttr_1h_media"],
        "Strato limite": ["sup_blh_media"],
        "Neve": ["sup_scaled_sf_1h_media"],
        "Quota": ["elev", "dz"],
        "Stagione": ["doy_sin", "doy_cos"],
    }
    if set(FORMA_RILIEVO) <= set(feature_cols(param)):
        g["Forma del rilievo"] = FORMA_RILIEVO
    coperte = sorted(c for v in g.values() for c in v)
    assert coperte == sorted(feature_cols(param)), set(feature_cols(param)) ^ set(coperte)
    return g


def rmse_con(net, xv, av, ov, param):
    with torch.no_grad():
        p = av + net(torch.tensor(xv, dtype=torch.float32)).numpy().ravel()
    if param in NON_NEGATIVI:
        p = np.clip(p, 0, None)
    return float(np.sqrt(np.mean((p - ov) ** 2)))


def main():
    torch.set_num_threads(6)
    epoche = json.load(open("epoche_cv.json"))
    t = carica_tabella()
    rng = np.random.default_rng(0)
    righe = []
    for param in PARAMS:
        feature = feature_cols(param)
        g = gruppi(param)
        tp = t[t["parameter"] == param].dropna(subset=feature)
        for reg, anno, tr, va in folds(tp):
            mean = tr[feature].values.mean(axis=0)
            std = tr[feature].values.std(axis=0)
            std[std == 0] = 1
            net = allena_rete(tr, feature, epoche[param], 0, mean, std)
            xv = (va[feature].values - mean) / std
            av, ov = va["aurora"].values, va["obs"].values
            base = rmse_con(net, xv, av, ov, param)
            candidati = [("feature", f, [f]) for f in feature] + [("gruppo", n, c) for n, c in g.items()]
            for tipo, nome, colonne in candidati:
                idx = [feature.index(c) for c in colonne]
                aumenti = []
                for _ in range(RIPETIZIONI):
                    xp = xv.copy()
                    xp[:, idx] = xv[rng.permutation(len(xv))][:, idx]   # stessa permutazione per il gruppo
                    aumenti.append(100 * (rmse_con(net, xp, av, ov, param) - base) / base)
                righe.append({"parametro": param, "regione": reg, "anno": anno, "tipo": tipo, "nome": nome,
                              "aumento_rmse_%": float(np.mean(aumenti)), "rmse_base": base})
            print(f"[{param}] {reg} {anno} fatto (RMSE {base:.3f})", flush=True)
            pd.DataFrame(righe).to_csv("esperimento_importanza_risultati.csv", index=False)

    res = pd.DataFrame(righe)
    m = res.groupby(["tipo", "nome", "parametro"])["aumento_rmse_%"].mean().unstack("parametro")[PARAMS]
    pd.set_option("display.width", 200)
    print("\n=== aumento % dell'RMSE mescolando un GRUPPO (media sui 9 fold) ===")
    gr = m.loc["gruppo"]
    print(gr.loc[gr.mean(axis=1).sort_values(ascending=False).index].round(2).to_string())
    print("\n=== aumento % dell'RMSE mescolando una SINGOLA feature (media sui 9 fold) ===")
    fe = m.loc["feature"]
    print(fe.loc[fe.mean(axis=1).sort_values(ascending=False).index].round(2).to_string())


if __name__ == "__main__":
    main()
