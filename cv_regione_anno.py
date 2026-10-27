"""
Cross-validation regione x anno del modello finale (train_downscaling_head.py): per ogni combinazione
(regione R, anno Y) il modello si allena sulle altre regioni negli altri anni e si valuta sulla
regione R nell'anno Y. Ogni fold riproduce la situazione del test finale (Piemonte 2020): regione e
anno mai visti. (La leave-one-region-out da sola, inutili/loro_cv.py, mette insieme gli stessi giorni
in regioni diverse e risulta un po' ottimista: il meteo e l'errore di Aurora di un giorno sono simili
in tutta la Pianura Padana.)

Regioni di validazione: Lombardia, Emilia-Romagna, Veneto. Le stazioni di altre regioni dentro l'area
(es. Trentino) restano sempre nel training dei fold, come nel modello finale. Il Piemonte (split
"test") non entra mai.

Stessa configurazione del modello finale (train_downscaling_head.py): feature per variabile, rete
64-64 con dropout 0.1, ensemble di N_SEED reti (seed 0..N_SEED-1), previsione = media delle reti.
L'errore dell'ensemble sul fold tenuto fuori si misura dopo ogni epoca; il numero di epoche scelto e'
quello con l'RMSE medio sui fold piu' basso. Lo legge train_downscaling_head.py (epoche_cv.json).

Uso:
    python cv_regione_anno.py
Output:
    cv_regione_anno_risultati.csv (parametro, regione, anno, epoca, rmse_corretto, bias_corretto, mae_corretto,
                                   rmse_grezzo, rmse_rete_singola, n)
    epoche_cv.json                (numero di epoche scelto per parametro)
"""
import json

import numpy as np
import pandas as pd
import torch

from cv_comune import PARAMS, allena_valuta, carica_tabella, errori, folds
from train_downscaling_head import DROPOUT, HIDDEN, N_SEED, feature_cols

MAX_EPOCHE = {"tx3": 20, "tn3": 20, "rr7": 20, "qq2": 45}


def main():
    torch.set_num_threads(6)
    t = carica_tabella()
    mancanti = sorted({c for p in PARAMS for c in feature_cols(p)} - set(t.columns))
    if mancanti:
        raise SystemExit(f"nella tabella mancano {mancanti}: rilancia il notebook §6.6")
    anni = sorted(t["anno"].unique())
    print(f"anni: {anni} | fold per parametro: 9 | reti per fold: {N_SEED}", flush=True)

    rows = []
    for param in PARAMS:
        tp = t[t["parameter"] == param].dropna(subset=feature_cols(param))
        for reg, anno, tr, va in folds(tp):
            ov, av = va["obs"].values, va["aurora"].values
            pred = np.stack([allena_valuta(tr, va, feature_cols(param), param, hidden=HIDDEN, dropout=DROPOUT,
                                           max_epoche=MAX_EPOCHE[param], seed=s, salva_pred=True)[1]
                             for s in range(N_SEED)])                                # seed x epoca x righe
            for ep in range(pred.shape[1]):
                e = errori(pred[:, ep].mean(axis=0), ov)
                rows.append({"parametro": param, "regione": reg, "anno": anno, "epoca": ep + 1,
                             "rmse_corretto": e["rmse"], "bias_corretto": e["bias"], "mae_corretto": e["mae"],
                             "rmse_grezzo": errori(av, ov)["rmse"],
                             "rmse_rete_singola": float(np.mean([errori(pred[s, ep], ov)["rmse"]
                                                                 for s in range(N_SEED)])),
                             "n": len(va)})
            print(f"[{param}] {reg} {anno}: train={len(tr)} val={len(va)} rmse grezzo={rows[-1]['rmse_grezzo']:.3f}",
                  flush=True)
            pd.DataFrame(rows).to_csv("cv_regione_anno_risultati.csv", index=False)

    res = pd.DataFrame(rows)
    curva = res.groupby(["parametro", "epoca"])["rmse_corretto"].mean().unstack("parametro")
    epoche = {p: int(curva[p].idxmin()) for p in PARAMS}
    json.dump(epoche, open("epoche_cv.json", "w"), indent=1)
    print(f"\nepoche scelte (salvate in epoche_cv.json): {epoche}")

    best = pd.concat([res[(res["parametro"] == p) & (res["epoca"] == e)] for p, e in epoche.items()])
    print("\n=== CV regione x anno, all'epoca scelta: media sui fold ===")
    s = best.groupby("parametro")[["rmse_grezzo", "rmse_rete_singola", "rmse_corretto", "bias_corretto",
                                   "mae_corretto"]].mean().loc[PARAMS]
    s["miglioramento_%"] = 100 * (s["rmse_corretto"] - s["rmse_grezzo"]) / s["rmse_grezzo"]
    s["std_tra_fold"] = best.groupby("parametro")["rmse_corretto"].std()
    print(s.round(3))
    print("\nper regione e anno (rmse corretto):")
    print(best.pivot_table(index=["parametro", "regione"], columns="anno", values="rmse_corretto").round(3))


if __name__ == "__main__":
    main()
