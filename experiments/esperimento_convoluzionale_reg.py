"""
Esperimento: testa convoluzionale REGOLARIZZATA (seguito di esperimento_convoluzionale.py, che
overfittava dopo 2-3 epoche e perdeva contro la testa attuale, soprattutto su qq2).

Stessa finestra 5 x 5 celle di Aurora centrata sulla stazione, stessa CV regione x anno, stesse
esclusioni. Cambia:
  - early stopping vero: dentro ogni fold il 15% delle stazioni e il 15% dei giorni (blocchi di 10) di
    training sono tenuti da parte; l errore interno si misura sulle stazioni escluse nei giorni esclusi;
    ci si ferma quando l'errore su quelle non migliora per PAZIENZA epoche e si tornano ai pesi
    migliori. L'errore riportato e' quello sul fold di validazione a quell'epoca: l'epoca non e'
    scelta guardando la validazione.
  - dropout sui canali della finestra (Dropout2d 0.2) e nella parte densa (0.3)
  - AdamW, weight decay 1e-3 (prima Adam 1e-4)
  - 16 filtri invece di 32, learning rate 5e-4 invece di 1e-3

Uso:
    python esperimento_convoluzionale_reg.py
Output:
    esperimento_convoluzionale_reg_risultati.csv (curva per epoca, con l'epoca di stop),
    riepilogo a schermo (confronto con la testa attuale e con la convoluzionale non regolarizzata)
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
    # stazioni nuove E giorni nuovi (blocchi di 10). Provato:
    #  - solo stazioni tenute da parte: la rete impara il meteo di quei giorni dalle stazioni vicine;
    #  - solo giorni tenuti da parte: la rete riconosce la stazione dalla finestra e ne impara la
    #    correzione specifica.
    # In entrambi i casi l'errore interno scendeva fino all'epoca 20-24, mentre sul fold di
    # validazione risaliva gia' dall'epoca 5.
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
    print(f"campi: {tuple(cubo.shape)}", flush=True)
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
                print(f"[{param}] {reg} {anno}: stop all'epoca {stop}, RMSE {r_stop:.3f} "
                      f"(minimo {min(r['rmse'] for r in curva):.3f}, Aurora {curva[0]['rmse_aurora']:.3f})", flush=True)
                pd.DataFrame(rows).to_csv("risultati/esperimento_convoluzionale_reg_risultati.csv", index=False)

    res = pd.DataFrame(rows)
    stop = res[res["epoca"] == res["epoca_stop"]].set_index(["parametro", "regione", "anno"])["rmse"]
    mlp = pd.read_csv("risultati/cv_regione_anno_risultati_modello_precedente.csv")  # testa con cui e' stato fatto il confronto
    ep_mlp = mlp.groupby(["parametro", "epoca"])["rmse_corretto"].mean().groupby("parametro").idxmin()
    attuale = pd.concat([mlp[(mlp["parametro"] == p) & (mlp["epoca"] == e)] for p, e in ep_mlp])
    attuale = attuale.set_index(["parametro", "regione", "anno"])["rmse_corretto"]
    conv = pd.read_csv("risultati/esperimento_convoluzionale_risultati.csv")
    ep_conv = conv.groupby(["parametro", "epoca"])["rmse"].mean().groupby("parametro").idxmin()
    conv = pd.concat([conv[(conv["parametro"] == p) & (conv["epoca"] == e)] for p, e in ep_conv])
    conv = conv.set_index(["parametro", "regione", "anno"])["rmse"]
    per_fold = pd.DataFrame({"conv_reg": stop, "conv": conv, "testa_attuale": attuale}).dropna()
    print("\n=== CV regione x anno, RMSE medio sui fold ===")
    print(per_fold.groupby("parametro").mean().round(3))
    print("\nfold in cui la convoluzionale regolarizzata batte la testa attuale:",
          (per_fold["conv_reg"] < per_fold["testa_attuale"]).groupby("parametro").sum().to_dict())
    print("\nper fold:")
    print(per_fold.round(2))


if __name__ == "__main__":
    main()
