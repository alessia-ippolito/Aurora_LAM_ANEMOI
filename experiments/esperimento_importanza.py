"""
Feature importance (permutation importance) of the final model, by single feature and by group.

In each fold of the region x year CV (cv_comune.py) a network is trained with the final configuration
(train_downscaling_head.py: features per variable, 64-64, dropout 0.1, epochs from epoche_cv.json).
Then, on the validation data of the fold, the values of one feature (or of all the features of a
group, with the same row permutation) are shuffled at random and the increase in RMSE is measured.
Importance = percentage increase of the RMSE, mean over RIPETIZIONI permutations and over the 9 folds.

By group as well as by single feature, because many features are almost copies of each other (e.g.
aurora = a_<variable>, temperature at 925 and 850 hPa with r = 0.98): shuffling only one of them, the
model still uses its twin and the importance looks low even if the information is essential;
shuffling the whole group measures how much the information matters.

Caution: the permutation creates unrealistic combinations (e.g. 30 degrees at the surface and 0 at
850 hPa). The importance says how much the model USES a piece of information, not that this
information is the physical cause of the Aurora error.

The group names are kept in Italian: they are the keys stored in the results file
(e.g. "Stagione" = season, "Quota" = elevation, "Umidita'" = humidity, "Forma del rilievo" = terrain shape).

Usage:
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
    epoche = json.load(open("modello/epoche_cv.json"))
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
                    xp[:, idx] = xv[rng.permutation(len(xv))][:, idx]   # same permutation for the whole group
                    aumenti.append(100 * (rmse_con(net, xp, av, ov, param) - base) / base)
                righe.append({"parametro": param, "regione": reg, "anno": anno, "tipo": tipo, "nome": nome,
                              "aumento_rmse_%": float(np.mean(aumenti)), "rmse_base": base})
            print(f"[{param}] {reg} {anno} done (RMSE {base:.3f})", flush=True)
            pd.DataFrame(righe).to_csv("risultati/esperimento_importanza_risultati.csv", index=False)

    res = pd.DataFrame(righe)
    m = res.groupby(["tipo", "nome", "parametro"])["aumento_rmse_%"].mean().unstack("parametro")[PARAMS]
    pd.set_option("display.width", 200)
    print("\n=== % increase of the RMSE when shuffling a GROUP (mean over the 9 folds) ===")
    gr = m.loc["gruppo"]
    print(gr.loc[gr.mean(axis=1).sort_values(ascending=False).index].round(2).to_string())
    print("\n=== % increase of the RMSE when shuffling a SINGLE feature (mean over the 9 folds) ===")
    fe = m.loc["feature"]
    print(fe.loc[fe.mean(axis=1).sort_values(ascending=False).index].round(2).to_string())


if __name__ == "__main__":
    main()
