"""
Experiment: can redundant features be removed without losing accuracy?
Choices guided by esperimento_importanza.py and by the correlations between features (|r| > 0.9).
Final configuration (64-64, dropout 0.1, terrain shape for tn3 and qq2), region x year CV of
cv_comune.py, 2 seeds; the comparison is with "D1 dropout 0.1" of esperimento_iperparametri.py (same
configuration with all the features, same seeds).

  M  minima (minimal)   removes a_<variable> (exact copy of "aurora") and snow (importance ~0)
  R  ridotta (reduced)  M + one feature for each highly correlated pair, the one that matters less alone:
                        atm_t_850 (r = 0.98 with atm_t_925), sup_2t_media (0.98 with a_tx3), tpi_20km (0.98 with dz)

Usage:
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
                print(f"[{param}] {nome} ({len(feature)} features) seed {seed} done", flush=True)
                pd.DataFrame(righe).to_csv("risultati/esperimento_feature_ridotte_risultati.csv", index=False)

    res = pd.DataFrame(righe)
    base = pd.read_csv("risultati/esperimento_iperparametri_risultati.csv")
    base = base[base["variante"] == "D1 dropout 0.1"].assign(variante="all features")
    tutto = pd.concat([base, res], ignore_index=True)
    s = riepilogo(tutto)
    tab = s["rmse"].unstack("parametro")[PARAMS]
    print("\n=== Region x year CV: mean RMSE over the folds and the 2 seeds, at the best epoch ===")
    print(tab.round(3).to_string())
    print("\n% difference from all features:")
    print((100 * (tab - tab.loc["all features"]) / tab.loc["all features"]).round(2).to_string())
    print("\nbest epoch:")
    print(s["epoca"].unstack("parametro")[PARAMS].to_string())


if __name__ == "__main__":
    main()
