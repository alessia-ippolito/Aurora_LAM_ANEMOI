"""
Esperimento (punto 5): piccola ricerca degli iperparametri della rete, una modifica alla volta
rispetto alla configurazione attuale (64-64, dropout 0, lr 1e-3, weight decay 1e-4, batch 4096).
Feature: quelle scelte con esperimento_terreno.py (forma del rilievo solo per tn3 e qq2).
CV regione x anno di cv_comune.py; un seed per configurazione, poi le configurazioni migliori
si ricontrollano con un secondo seed (CONFERMA).

Una modifica alla volta (e non una griglia completa) per due motivi: costa meno e, provando poche
configurazioni sugli stessi 9 fold, si riduce il rischio di scegliere quella che va meglio per caso.

Uso:
    python esperimento_iperparametri.py
    python esperimento_iperparametri.py --conferma "H2 128-128" "W wd 1e-3"   # secondo seed
Output:
    esperimento_iperparametri_risultati.csv
"""
import sys

import pandas as pd
import torch

from cv_comune import PARAMS, carica_tabella, cv_rete, riepilogo
from descrittori_terreno import USCITA
from experiments.esperimento_terreno import FORMA
from train_downscaling_head import FEATURE_COLS

FEATURE = {p: FEATURE_COLS + (FORMA if p in ("tn3", "qq2") else []) for p in PARAMS}
BASE = dict(hidden=(64, 64), dropout=0.0, lr=1e-3, wd=1e-4, batch=4096, max_epoche=40)
CONFIGURAZIONI = {
    "B base 64-64": {},
    "H1 32-32": dict(hidden=(32, 32)),
    "H2 128-128": dict(hidden=(128, 128)),
    "H3 64-64-64": dict(hidden=(64, 64, 64)),
    "D1 dropout 0.1": dict(dropout=0.1),
    "D2 128-128 dropout 0.2": dict(hidden=(128, 128), dropout=0.2),
    "L lr 3e-4": dict(lr=3e-4, max_epoche=60),
    "W wd 1e-3": dict(wd=1e-3),
    "S batch 1024": dict(batch=1024),
}
USCITA_CSV = "risultati/esperimento_iperparametri_risultati.csv"


def main():
    torch.set_num_threads(6)
    conferma = sys.argv[sys.argv.index("--conferma") + 1:] if "--conferma" in sys.argv else None
    t = carica_tabella(colonne_extra=pd.read_parquet(USCITA))
    try:
        righe = pd.read_csv(USCITA_CSV).to_dict("records")
    except FileNotFoundError:
        righe = []
    seed = 1 if conferma else 0
    for nome in (conferma or CONFIGURAZIONI):
        opz = {**BASE, **CONFIGURAZIONI[nome]}
        for param in PARAMS:
            for r in cv_rete(t, param, FEATURE[param], nome, seed=seed, **opz):
                righe.append({**r, "seed": seed})
            print(f"{nome} [{param}] seed {seed} fatto", flush=True)
            pd.DataFrame(righe).to_csv(USCITA_CSV, index=False)

    res = pd.DataFrame(righe)
    for s in sorted(res["seed"].unique()):
        print(f"\n=== seed {s}: RMSE medio sui 9 fold all'epoca migliore ===")
        r = riepilogo(res[res["seed"] == s])
        tab = r["rmse"].unstack("parametro")[list(PARAMS)]
        base = tab.loc["B base 64-64"]
        print(tab.round(3).to_string())
        print("\ndifferenza % rispetto alla base:")
        print((100 * (tab - base) / base).round(2).to_string())
        print("\nepoca migliore:")
        print(r["epoca"].unstack("parametro")[list(PARAMS)].to_string())


if __name__ == "__main__":
    main()
