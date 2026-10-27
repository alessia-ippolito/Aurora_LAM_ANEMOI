"""
Esperimento: pioggia e radiazione di Aurora interpolate con IDW invece che bilinearmente nel punto della
stazione. Stessa cross-validation, stessa rete e stessi seed di esperimento_bilineare.py; cambia solo come
si interpolano rr7 e qq2 (tx3, tn3 e le altre variabili restano bilineari).

Varianti:
  B bilineare:   come in esperimento_bilineare.py
  C bilineare+:  come in esperimento_bilineare.py
  D idw:         B con rr7 e qq2 di Aurora interpolati con IDW (4 celle piu' vicine, pesi 1/d^2)
  E idw+:        C con a_rr7 e a_qq2 (target e feature) interpolati con IDW
Per tx3 e tn3 D coincide con B; E differisce da C solo nelle feature a_rr7 e a_qq2.

Uso:
    python esperimento_idw.py
Output:
    esperimento_idw_risultati.csv, riepilogo a schermo
"""
from esperimento_bilineare import BASE, EXTRA, main

EXTRA_IDW = [f"{c}_idw" if c in ("a_rr7", "a_qq2") else c for c in EXTRA]
VARIANTI = {"B bilineare": ("aurora_B", BASE), "C bilineare+": ("aurora_B", BASE + EXTRA),
            "D idw": ("aurora_D", BASE), "E idw+": ("aurora_D", BASE + EXTRA_IDW)}

if __name__ == "__main__":
    main(VARIANTI, "esperimento_idw_risultati.csv", idw=True)
