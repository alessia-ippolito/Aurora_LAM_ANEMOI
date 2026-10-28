"""
Experiment: REGULARISED convolutional head (follow-up of esperimento_convoluzionale.py, which
overfitted after 2-3 epochs and lost against the current head, especially on qq2).

Same 5 x 5 window of Aurora cells centred on the station, same region x year CV, same exclusions.
What changes:
  - true early stopping: inside each fold 15% of the stations and 15% of the training days (blocks
    of 10) are held out; the internal error is measured on the held-out stations on the held-out
    days; training stops when the error on those does not improve for PAZIENZA (patience) epochs and
    the best weights are restored. The reported error is the one on the validation fold at that
    epoch: the epoch is not chosen by looking at the validation.
  - dropout on the window channels (Dropout2d 0.2) and in the dense part (0.3)
  - AdamW, weight decay 1e-3 (before: Adam 1e-4)
  - 16 filters instead of 32, learning rate 5e-4 instead of 1e-3

Usage:
    python esperimento_convoluzionale_reg.py
Output:
    esperimento_convoluzionale_reg_risultati.csv (curve per epoch, with the stopping epoch),
    summary on screen (comparison with the current head and with the non-regularised convolutional one)
"""
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from experiments.esperimento_convoluzionale import (NON_NEGATIVI, SCALARI, TABLE, cubo_aurora, finestre, pesi_finestra)
from cv_comune import BATCH, PARAMS, REGIONS, SEED, assign_regions
from train_downscaling_head import applica_esclusioni

MAX_EPOCHS = 40
PAZIENZA = 3
QUOTA_INTERNA = 0.15
LR = 5e-4
WEIGHT_DECAY = 1e-3


class TestaConvReg(nn.Module):
    def __init__(self, canali, n_scalari, k=16, h=64, p_canali=0.2, p_densa=0.3):
        super().__init__()
        self.conv = nn.Sequential(nn.Dropout2d(p_canali), nn.Conv2d(canali, k, 3), nn.ReLU(),
                                  nn.Conv2d(k, k, 3), nn.ReLU())
        self.mlp = nn.Sequential(nn.Linear(k + n_scalari, h), nn.ReLU(), nn.Dropout(p_densa),
                                 nn.Linear(h, h), nn.ReLU(), nn.Dropout(p_densa), nn.Linear(h, 1))

    def forward(self, finestra, scalari):
        return self.mlp(torch.cat([self.conv(finestra).flatten(1), scalari], dim=1))


def rmse_su(net, cubo, pesi, g, st, sc, aurora, obs, taglia_zero):
    net.eval()
    with torch.no_grad():
        p = torch.cat([net(finestre(cubo, g[i:i + BATCH], st[i:i + BATCH], *pesi), sc[i:i + BATCH])
                       for i in range(0, len(g), BATCH)]).numpy().ravel()
    corr = aurora + p
    if taglia_zero:
        corr = np.clip(corr, 0, None)
    e = corr - obs
    return float(np.sqrt(np.mean(e ** 2))), float(e.mean())


def fold(cubo, pesi, tr, va, taglia_zero, seed=SEED):
    torch.manual_seed(seed)
    # new stations AND new days (blocks of 10). Tried:
    #  - held-out stations only: the network learns the weather of those days from the nearby stations;
    #  - held-out days only: the network recognises the station from the window and learns its
    #    specific correction.
    # In both cases the internal error kept falling until epoch 20-24, while on the validation fold
    # it was already rising from epoch 5.
    rng = np.random.default_rng(seed)
    uid = tr["uid"].unique()
    st_int = rng.choice(uid, int(len(uid) * QUOTA_INTERNA), replace=False)
    blocco = tr["giorno_i"] // 10
    blocchi = blocco.unique()
    gi_int = rng.choice(blocchi, int(len(blocchi) * QUOTA_INTERNA), replace=False)
    s_in, g_in = tr["uid"].isin(st_int), blocco.isin(gi_int)
    es = tr[s_in & g_in]
    tr = tr[~s_in & ~g_in]
    m = tr[SCALARI].values.mean(axis=0)
    s = tr[SCALARI].values.std(axis=0)
    s[s == 0] = 1

    def tensori(df):
        return (torch.tensor(df["giorno_i"].values), torch.tensor(df["staz_i"].values),
                torch.tensor((df[SCALARI].values - m) / s, dtype=torch.float32),
                df["aurora"].values, df["obs"].values)

    g, st, sc, _, _ = tensori(tr)
    y = torch.tensor(tr["residual"].values, dtype=torch.float32).unsqueeze(1)
    T_es, T_va = tensori(es), tensori(va)
    net = TestaConvReg(cubo.shape[1], len(SCALARI))
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    out, migliore, stato, attesa = [], np.inf, None, 0
    for ep in range(1, MAX_EPOCHS + 1):
        net.train()
        perm = torch.randperm(len(y))
        for i in range(0, len(y), BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            torch.nn.functional.mse_loss(net(finestre(cubo, g[b], st[b], *pesi), sc[b]), y[b]).backward()
            opt.step()
        r_es, _ = rmse_su(net, cubo, pesi, *T_es, taglia_zero)
        r_va, b_va = rmse_su(net, cubo, pesi, *T_va, taglia_zero)
        out.append({"epoca": ep, "rmse_interna": r_es, "rmse": r_va, "bias": b_va,
                    "rmse_aurora": float(np.sqrt(np.mean((T_va[3] - T_va[4]) ** 2))), "n": len(va)})
        if r_es < migliore - 1e-4:
            migliore, stato, attesa = r_es, ep, 0
        else:
            attesa += 1
            if attesa >= PAZIENZA:
                break
    for r in out:
        r["epoca_stop"] = stato
    return out


def main():
    torch.set_num_threads(6)
    t = pd.read_parquet(TABLE)
    t = applica_esclusioni(t[t["split"] == "train"])
    doy = t["date"].dt.dayofyear.astype(float)
    t["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    t["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    t["anno"] = t["date"].dt.year
    t = assign_regions(t)
    anni = sorted(t["anno"].unique())

    cubo, giorni, glat, glon, nomi = cubo_aurora(anni)
    print(f"fields: {tuple(cubo.shape)}", flush=True)
    st = t.drop_duplicates("uid")[["uid", "lat", "lon"]].reset_index(drop=True)
    pesi = pesi_finestra(st["lat"].values, st["lon"].values, glat, glon)
    t["staz_i"] = t["uid"].map(pd.Series(st.index, index=st["uid"]))
    t["giorno_i"] = giorni.get_indexer(t["date"])
    t = t[t["giorno_i"] >= 0]

    rows = []
    for param in PARAMS:
        tp = t[t["parameter"] == param]
        for reg in REGIONS:
            for anno in anni:
                tr = tp[(tp["regione"] != reg) & (tp["anno"] != anno)]
                va = tp[(tp["regione"] == reg) & (tp["anno"] == anno)]
                if tr.empty or va.empty:
                    continue
                curva = fold(cubo, pesi, tr, va, param in NON_NEGATIVI)
                rows += [{"parametro": param, "regione": reg, "anno": anno, **r} for r in curva]
                stop = curva[0]["epoca_stop"]
                r_stop = next(r["rmse"] for r in curva if r["epoca"] == stop)
                print(f"[{param}] {reg} {anno}: stop at epoch {stop}, RMSE {r_stop:.3f} "
                      f"(minimum {min(r['rmse'] for r in curva):.3f}, Aurora {curva[0]['rmse_aurora']:.3f})", flush=True)
                pd.DataFrame(rows).to_csv("risultati/esperimento_convoluzionale_reg_risultati.csv", index=False)

    res = pd.DataFrame(rows)
    stop = res[res["epoca"] == res["epoca_stop"]].set_index(["parametro", "regione", "anno"])["rmse"]
    mlp = pd.read_csv("risultati/cv_regione_anno_risultati_modello_precedente.csv")  # head the comparison was made with
    ep_mlp = mlp.groupby(["parametro", "epoca"])["rmse_corretto"].mean().groupby("parametro").idxmin()
    attuale = pd.concat([mlp[(mlp["parametro"] == p) & (mlp["epoca"] == e)] for p, e in ep_mlp])
    attuale = attuale.set_index(["parametro", "regione", "anno"])["rmse_corretto"]
    conv = pd.read_csv("risultati/esperimento_convoluzionale_risultati.csv")
    ep_conv = conv.groupby(["parametro", "epoca"])["rmse"].mean().groupby("parametro").idxmin()
    conv = pd.concat([conv[(conv["parametro"] == p) & (conv["epoca"] == e)] for p, e in ep_conv])
    conv = conv.set_index(["parametro", "regione", "anno"])["rmse"]
    per_fold = pd.DataFrame({"conv_reg": stop, "conv": conv, "testa_attuale": attuale}).dropna()
    print("\n=== Region x year CV, mean RMSE over the folds ===")
    print(per_fold.groupby("parametro").mean().round(3))
    print("\nfolds where the regularised convolutional head beats the current head:",
          (per_fold["conv_reg"] < per_fold["testa_attuale"]).groupby("parametro").sum().to_dict())
    print("\nby fold:")
    print(per_fold.round(2))


if __name__ == "__main__":
    main()
