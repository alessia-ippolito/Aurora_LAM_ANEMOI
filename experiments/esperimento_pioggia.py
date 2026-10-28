"""
Experiment (point 3): is precipitation (rr7) corrected better by changing how the network learns it?
Same features as the current model, region x year CV of cv_comune.py, 2 seeds per variant.

  P0 residuo (residual)  as today: the network learns obs - aurora with squared error
  P1 radice (sqrt)       the network learns sqrt(obs) - sqrt(aurora): gives less weight to the few extreme events
  P2 log                 the network learns log(1+obs) - log(1+aurora): like P1, even more marked
  P3 due parti (two-part) one network estimates the probability of a wet day (>= 1 mm), a second one
                         the amount on wet days; forecast = probability x amount

With squared error P0 already estimates the conditional mean, which is what minimises the RMSE:
P1-P3 can improve other aspects (absolute error, recognising rainy days) even if the RMSE does not
drop. This is why, besides RMSE, bias and MAE, the following are measured for the 1 and 20 mm thresholds:
  POD  fraction of the days with precipitation >= threshold that are detected
  FAR  fraction of the days forecast >= threshold on which it did not rain that much
  CSI  overall index (hits / (hits + misses + false alarms))

Usage:
    python esperimento_pioggia.py
Output:
    esperimento_pioggia_risultati.csv
"""
import numpy as np
import pandas as pd
import torch

from cv_comune import BATCH, Rete, allena_valuta, carica_tabella, errori, folds
from train_downscaling_head import FEATURE_COLS

SEEDS = [0, 1]
MAX_EPOCHE = 25
SOGLIE = [1.0, 20.0]
SOGLIA_PIOVOSO = 1.0


def categoriche(pred, obs):
    out = {}
    for s in SOGLIE:
        sp, so = pred >= s, obs >= s
        colpi, mancati, falsi = (sp & so).sum(), (~sp & so).sum(), (sp & ~so).sum()
        out[f"pod_{s:g}"] = colpi / max(colpi + mancati, 1)
        out[f"far_{s:g}"] = falsi / max(colpi + falsi, 1)
        out[f"csi_{s:g}"] = colpi / max(colpi + mancati + falsi, 1)
    return out


def due_parti(tr, va, seed):
    """Classifier (wet day yes/no) + regression of the amount on wet days only."""
    torch.manual_seed(seed)
    m = tr[FEATURE_COLS].values.mean(axis=0)
    s = tr[FEATURE_COLS].values.std(axis=0)
    s[s == 0] = 1
    X = lambda d: torch.tensor((d[FEATURE_COLS].values - m) / s, dtype=torch.float32)
    x, xv = X(tr), X(va)
    piov = torch.tensor((tr["obs"].values >= SOGLIA_PIOVOSO).astype("float32")).unsqueeze(1)
    res = torch.tensor(tr["residual"].values, dtype=torch.float32).unsqueeze(1)
    bagnati = torch.where(piov.squeeze(1) > 0)[0]
    cls, reg = Rete(len(FEATURE_COLS)), Rete(len(FEATURE_COLS))
    oc = torch.optim.Adam(cls.parameters(), lr=1e-3, weight_decay=1e-4)
    orr = torch.optim.Adam(reg.parameters(), lr=1e-3, weight_decay=1e-4)
    av = va["aurora"].values
    preds = []
    for _ in range(MAX_EPOCHE):
        cls.train(); reg.train()
        perm = torch.randperm(len(x))
        for i in range(0, len(x), BATCH):
            b = perm[i:i + BATCH]
            oc.zero_grad()
            torch.nn.functional.binary_cross_entropy_with_logits(cls(x[b]), piov[b]).backward()
            oc.step()
        perm = bagnati[torch.randperm(len(bagnati))]
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            orr.zero_grad()
            torch.nn.functional.mse_loss(reg(x[b]), res[b]).backward()
            orr.step()
        cls.eval(); reg.eval()
        with torch.no_grad():
            p = torch.sigmoid(cls(xv)).numpy().ravel()
            q = np.clip(av + reg(xv).numpy().ravel(), 0, None)
        preds.append((p * q).astype("float32"))
    return np.array(preds)


def main():
    torch.set_num_threads(6)
    t = carica_tabella()
    tp = t[t["parameter"] == "rr7"]
    righe = []
    for reg, anno, tr, va in folds(tp):
        ov = va["obs"].values
        for seed in SEEDS:
            varianti = {}
            for nome, trasf in [("P0 residuo", "nessuna"), ("P1 radice", "radice"), ("P2 log", "log1p")]:
                _, varianti[nome] = allena_valuta(tr, va, FEATURE_COLS, "rr7", max_epoche=MAX_EPOCHE, seed=seed,
                                                  trasformazione=trasf, salva_pred=True)
            varianti["P3 due parti"] = due_parti(tr, va, seed)
            for nome, pred in varianti.items():
                for ep, p in enumerate(pred, start=1):
                    righe.append({"variante": nome, "regione": reg, "anno": anno, "seed": seed, "epoca": ep,
                                  **errori(p, ov), **categoriche(p, ov)})
        print(f"{reg} {anno} done", flush=True)
        pd.DataFrame(righe).to_csv("risultati/esperimento_pioggia_risultati.csv", index=False)

    res = pd.DataFrame(righe)
    # epoch chosen per variant on the mean RMSE (as for the model), then all the metrics at that epoch
    curva = res.groupby(["variante", "epoca"]).mean(numeric_only=True)
    ep = curva["rmse"].groupby("variante").idxmin()
    tab = pd.DataFrame({v: curva.loc[e] for v, e in ep.items()}).T.drop(columns=["anno", "seed"])
    tab.insert(0, "epoca", [e[1] for e in ep])
    rif = []
    for reg, anno, _, va in folds(tp):
        rif.append({**errori(va["aurora"].values, va["obs"].values), **categoriche(va["aurora"].values, va["obs"].values)})
    tab.loc["Aurora interpolated"] = pd.DataFrame(rif).mean()
    pd.set_option("display.width", 200)
    print("\n=== rr7, region x year CV: means over folds and seeds, at the epoch with the best RMSE ===")
    print(tab.round(3).to_string())


if __name__ == "__main__":
    main()
