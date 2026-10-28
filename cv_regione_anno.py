"""
Region x year cross-validation of the final model (train_downscaling_head.py): for each combination
(region R, year Y) the model is trained on the other regions in the other years and evaluated on
region R in year Y. Each fold reproduces the situation of the final test (Piedmont 2020): a region
and a year never seen. (Leave-one-region-out alone, inutili/loro_cv.py, keeps the same days in
different regions together and turns out a little optimistic: the weather and the Aurora error of a
day are similar over the whole Po Valley.)

Validation regions: Lombardia, Emilia-Romagna, Veneto. Stations of other regions inside the area
(e.g. Trentino) always stay in the training part of the folds, as in the final model. Piedmont
(split "test") is never used.

Same configuration as the final model (train_downscaling_head.py): features per variable, 64-64
network with dropout 0.1, ensemble of N_SEED networks (seeds 0..N_SEED-1), forecast = mean of the
networks. The ensemble error on the held-out fold is measured after each epoch; the chosen number of
epochs is the one with the lowest mean RMSE over the folds. It is read by train_downscaling_head.py
(epoche_cv.json).

Usage:
    python cv_regione_anno.py
Output:
    cv_regione_anno_risultati.csv (parametro, regione, anno, epoca, rmse_corretto, bias_corretto, mae_corretto,
                                   rmse_grezzo, rmse_rete_singola, n)
    epoche_cv.json                (number of epochs chosen per variable)
"""
import json

import numpy as np
import pandas as pd
import torch

from cv_comune import PARAMS, allena_valuta, carica_tabella, errori, folds
from train_downscaling_head import DROPOUT, HIDDEN, N_SEED, feature_cols

EPOCHE_CV = "modello/epoche_cv.json"  # read by train_downscaling_head.py
MAX_EPOCHE = {"tx3": 20, "tn3": 20, "rr7": 20, "qq2": 45}


def main():
    torch.set_num_threads(6)
    t = carica_tabella()
    mancanti = sorted({c for p in PARAMS for c in feature_cols(p)} - set(t.columns))
    if mancanti:
        raise SystemExit(f"the table is missing {mancanti}: re-run the notebook §6.6")
    anni = sorted(t["anno"].unique())
    print(f"years: {anni} | folds per variable: 9 | networks per fold: {N_SEED}", flush=True)

    rows = []
    for param in PARAMS:
        tp = t[t["parameter"] == param].dropna(subset=feature_cols(param))
        for reg, anno, tr, va in folds(tp):
            ov, av = va["obs"].values, va["aurora"].values
            pred = np.stack([allena_valuta(tr, va, feature_cols(param), param, hidden=HIDDEN, dropout=DROPOUT,
                                           max_epoche=MAX_EPOCHE[param], seed=s, salva_pred=True)[1]
                             for s in range(N_SEED)])                                # seed x epoch x rows
            for ep in range(pred.shape[1]):
                e = errori(pred[:, ep].mean(axis=0), ov)
                rows.append({"parametro": param, "regione": reg, "anno": anno, "epoca": ep + 1,
                             "rmse_corretto": e["rmse"], "bias_corretto": e["bias"], "mae_corretto": e["mae"],
                             "rmse_grezzo": errori(av, ov)["rmse"],
                             "rmse_rete_singola": float(np.mean([errori(pred[s, ep], ov)["rmse"]
                                                                 for s in range(N_SEED)])),
                             "n": len(va)})
            print(f"[{param}] {reg} {anno}: train={len(tr)} val={len(va)} raw rmse={rows[-1]['rmse_grezzo']:.3f}",
                  flush=True)
            pd.DataFrame(rows).to_csv("risultati/cv_regione_anno_risultati.csv", index=False)

    res = pd.DataFrame(rows)
    curva = res.groupby(["parametro", "epoca"])["rmse_corretto"].mean().unstack("parametro")
    epoche = {p: int(curva[p].idxmin()) for p in PARAMS}
    json.dump(epoche, open(EPOCHE_CV, "w"), indent=1)
    print(f"\nchosen epochs (saved in epoche_cv.json): {epoche}")

    best = pd.concat([res[(res["parametro"] == p) & (res["epoca"] == e)] for p, e in epoche.items()])
    print("\n=== Region x year CV, at the chosen epoch: mean over the folds ===")
    s = best.groupby("parametro")[["rmse_grezzo", "rmse_rete_singola", "rmse_corretto", "bias_corretto",
                                   "mae_corretto"]].mean().loc[PARAMS]
    s["miglioramento_%"] = 100 * (s["rmse_corretto"] - s["rmse_grezzo"]) / s["rmse_grezzo"]
    s["std_tra_fold"] = best.groupby("parametro")["rmse_corretto"].std()
    print(s.round(3))
    print("\nby region and year (corrected rmse):")
    print(best.pivot_table(index=["parametro", "regione"], columns="anno", values="rmse_corretto").round(3))


if __name__ == "__main__":
    main()
