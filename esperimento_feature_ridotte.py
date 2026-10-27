"""
Esperimento: si possono togliere feature ridondanti senza perdere accuratezza?
Scelte guidate da esperimento_importanza.py e dalle correlazioni tra feature (|r| > 0.9).
Configurazione finale (64-64, dropout 0.1, forma del rilievo per tn3 e qq2), CV regione x anno di
cv_comune.py, 2 seed; il confronto e' con "D1 dropout 0.1" di esperimento_iperparametri.py (stessa
configurazione con tutte le feature, stessi seed).

  M  minima     toglie a_<variabile> (copia esatta di "aurora") e la neve (importanza ~0)
  R  ridotta    M + una feature per ogni coppia molto correlata, quella che conta meno da sola:
                atm_t_850 (r = 0.98 con atm_t_925), sup_2t_media (0.98 con a_tx3), tpi_20km (0.98 con dz)

Uso:
    python esperimento_feature_ridotte.py
Output:
    esperimento_feature_ridotte_risultati.csv
"""
import pandas as pd
import torch

from cv_comune import PARAMS, carica_tabella, cv_rete, riepilogo
from train_downscaling_head import feature_cols

SEEDS = [0, 1]
TOGLI = {
    "M minima": lambda p: [f"a_{p}", "sup_scaled_sf_1h_media"],
    "R ridotta": lambda p: [f"a_{p}", "sup_scaled_sf_1h_media", "atm_t_850", "sup_2t_media", "tpi_20km"],
}


def main():
    torch.set_num_threads(6)
    t = carica_tabella()
    righe = []
    for param in PARAMS:
        for nome, togli in TOGLI.items():
            feature = [c for c in feature_cols(param) if c not in togli(param)]
            for seed in SEEDS:
                for r in cv_rete(t, param, feature, nome, seed=seed, dropout=0.1):
                    righe.append({**r, "seed": seed})
                print(f"[{param}] {nome} ({len(feature)} feature) seed {seed} fatto", flush=True)
                pd.DataFrame(righe).to_csv("esperimento_feature_ridotte_risultati.csv", index=False)

    res = pd.DataFrame(righe)
    base = pd.read_csv("esperimento_iperparametri_risultati.csv")
    base = base[base["variante"] == "D1 dropout 0.1"].assign(variante="tutte le feature")
    tutto = pd.concat([base, res], ignore_index=True)
    s = riepilogo(tutto)
    tab = s["rmse"].unstack("parametro")[PARAMS]
    print("\n=== CV regione x anno: RMSE medio sui fold e sui 2 seed, all'epoca migliore ===")
    print(tab.round(3).to_string())
    print("\ndifferenza % rispetto a tutte le feature:")
    print((100 * (tab - tab.loc["tutte le feature"]) / tab.loc["tutte le feature"]).round(2).to_string())
    print("\nepoca migliore:")
    print(s["epoca"].unstack("parametro")[PARAMS].to_string())


if __name__ == "__main__":
    main()
