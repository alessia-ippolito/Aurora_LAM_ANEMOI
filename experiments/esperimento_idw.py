"""
Experiment: Aurora precipitation and radiation interpolated with IDW instead of bilinearly at the
station location. Same cross-validation, same network and same seeds as esperimento_bilineare.py; only
the interpolation of rr7 and qq2 changes (tx3, tn3 and the other variables stay bilinear).

Variants:
  B bilineare:   as in esperimento_bilineare.py
  C bilineare+:  as in esperimento_bilineare.py
  D idw:         B with Aurora rr7 and qq2 interpolated with IDW (4 nearest cells, weights 1/d^2)
  E idw+:        C with a_rr7 and a_qq2 (target and features) interpolated with IDW
For tx3 and tn3 D coincides with B; E differs from C only in the features a_rr7 and a_qq2.

Usage:
    python esperimento_idw.py
Output:
    esperimento_idw_risultati.csv, summary on screen
"""
from experiments.esperimento_bilineare import BASE, EXTRA, main

EXTRA_IDW = [f"{c}_idw" if c in ("a_rr7", "a_qq2") else c for c in EXTRA]
VARIANTI = {"B bilineare": ("aurora_B", BASE), "C bilineare+": ("aurora_B", BASE + EXTRA),
            "D idw": ("aurora_D", BASE), "E idw+": ("aurora_D", BASE + EXTRA_IDW)}

if __name__ == "__main__":
    main(VARIANTI, "risultati/esperimento_idw_risultati.csv", idw=True)
