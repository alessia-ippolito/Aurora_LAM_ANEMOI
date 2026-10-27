"""
Esperimento: testa convoluzionale. Oltre ai valori di Aurora nel punto della stazione, la rete vede
una finestra di 5 x 5 celle di Aurora (~125 x 125 km) centrata sul punto esatto della stazione, con
tutte le variabili usate dalla testa attuale: puo' riconoscere la situazione intorno (fronti, flusso
contro i rilievi, differenze pianura/montagna, nuvolosita' a monte).

Confronto diretto con la testa attuale (MLP sulle 28 feature nel punto, cv_regione_anno.py):
stessa cross-validation regione x anno (allena su 2 regioni x 2 anni, valuta su regione e anno
esclusi), stesse esclusioni (notebook §7.2), stessa tabella (training_table_aurora.parquet),
Piemonte mai usato. L'errore si misura dopo ogni epoca; si sceglie l'epoca migliore in media.

Finestra: celle campionate con interpolazione bilineare a passi di 0.25 gradi intorno al punto
(offset -0.5 ... +0.5 gradi in lat e lon), quindi centrata esattamente sulla stazione (o sul pixel,
nelle mappe). I campi hanno 0.5 gradi di margine oltre l'AOI: bastano per stazioni sul bordo.

Rete:  finestra (24 canali x 5 x 5) -> conv 3x3 (32) -> conv 3x3 (32) -> 32 numeri
       + valori nel punto (aurora, quota, dz, stagione) -> 64 -> 64 -> correzione

Uso:
    python esperimento_convoluzionale.py        # finestra 5 x 5
    python esperimento_convoluzionale.py 7      # finestra 7 x 7 (+-0.75 gradi: oltre il margine dei
                                                  campi per le stazioni vicino al bordo dell'AOI, dove
                                                  la finestra ripete i valori del bordo)
Output:
    esperimento_convoluzionale_risultati.csv (esperimento_convoluzionale_7x7_risultati.csv per 7 x 7), riepilogo a schermo (con il confronto con la testa attuale)
"""
import json
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from campi_aurora import carica_campi
from cv_comune import BATCH, PARAMS, REGIONS, SEED, assign_regions
from train_downscaling_head import applica_esclusioni

TABLE = "tabelle/training_table_aurora.parquet"
INFO = json.load(open("tabelle/fattori_scala_aurora.json"))
MAX_EPOCHS = 25
COL_AURORA = {"tx3": "tx3_aurora", "tn3": "tn3_aurora", "rr7": "rr7_aurora_raw", "qq2": "qq2_aurora_raw"}
NON_NEGATIVI = {"rr7", "qq2"}
SUP = ["sup_2t_media", "sup_msl_media", "sup_10u_media", "sup_10v_media", "sup_vento10_max",
       "sup_i10fg_media", "sup_blh_media", "sup_ttr_1h_media", "sup_scaled_sf_1h_media"]
ATM = [("atm_t_media", 925), ("atm_t_media", 850), ("atm_t_media", 700), ("atm_t_media", 500),
       ("atm_q_media", 925), ("atm_q_media", 850), ("atm_q_media", 700), ("atm_z_media", 500),
       ("atm_u_media", 850), ("atm_v_media", 850)]
SCALARI = ["aurora", "elev", "dz", "doy_sin", "doy_cos"]
LATO = int(sys.argv[1]) if __name__ == "__main__" and len(sys.argv) > 1 else 5   # celle per lato (dispari)
OFFSET = 0.25 * (np.arange(LATO) - LATO // 2)
RISULTATI = ("risultati/esperimento_convoluzionale_risultati.csv" if LATO == 5
             else f"risultati/esperimento_convoluzionale_{LATO}x{LATO}_risultati.csv")


def cubo_aurora(anni):
    """Campi giornalieri come un unico array (giorni, canali, lat, lon), canali standardizzati."""
    campi = carica_campi(anni, variabili=list(COL_AURORA.values()) + SUP + ["orog_aurora"]
                         + sorted({v for v, _ in ATM}))
    fatt, qc = INFO["fattori_scala"], {k: tuple(v) for k, v in INFO["table_qc"].items()}
    canali, nomi = [], []
    for p, c in COL_AURORA.items():
        canali.append((campi[c] * fatt[p]).clip(*qc[p]).values); nomi.append(f"a_{p}")
    for v in SUP:
        canali.append(campi[v].values); nomi.append(v)
    for v, lv in ATM:
        canali.append(campi[v].sel(livello=lv).values); nomi.append(f"{v}_{lv}")
    orog = np.broadcast_to(campi["orog_aurora"].values, canali[0].shape)
    canali.append(orog); nomi.append("orog_aurora")
    cubo = np.stack(canali, axis=1).astype("float32")                        # D x C x H x W
    media = cubo.mean(axis=(0, 2, 3), keepdims=True)
    dev = cubo.std(axis=(0, 2, 3), keepdims=True)
    dev[dev == 0] = 1
    cubo = (cubo - media) / dev
    giorni = pd.DatetimeIndex(campi["date"].values)
    return torch.tensor(cubo), giorni, campi["lat"].values, campi["lon"].values, nomi


def pesi_finestra(lat_st, lon_st, glat, glon):
    """Per ogni stazione: indici delle 4 celle e pesi bilineari per i LATO x LATO punti della finestra."""
    dlat, dlon = glat[0] - glat[1], glon[1] - glon[0]     # lat decrescente, lon crescente
    la, lo = np.broadcast_arrays(lat_st[:, None, None] + OFFSET[None, :, None],
                                 lon_st[:, None, None] + OFFSET[None, None, :])
    fy = (glat[0] - la) / dlat
    fx = (lo - glon[0]) / dlon
    y0 = np.clip(np.floor(fy), 0, len(glat) - 2).astype(int)
    x0 = np.clip(np.floor(fx), 0, len(glon) - 2).astype(int)
    wy = np.clip(fy - y0, 0, 1)
    wx = np.clip(fx - x0, 0, 1)
    f = lambda a: torch.tensor(a.reshape(len(lat_st), LATO * LATO))
    return f(y0), f(x0), f(wy).float(), f(wx).float()


def finestre(cubo, giorno, staz, y0, x0, wy, wx):
    """Finestre (B, C, LATO, LATO) per i campioni (giorno, stazione)."""
    yy, xx, ay, ax = y0[staz], x0[staz], wy[staz], wx[staz]                  # B x 25
    d = giorno[:, None].expand_as(yy)
    v = lambda dy, dx: cubo[d, :, yy + dy, xx + dx]                          # B x 25 x C
    p = ((1 - ay)[..., None] * (1 - ax)[..., None] * v(0, 0) + (1 - ay)[..., None] * ax[..., None] * v(0, 1)
         + ay[..., None] * (1 - ax)[..., None] * v(1, 0) + ay[..., None] * ax[..., None] * v(1, 1))
    return p.permute(0, 2, 1).reshape(len(giorno), cubo.shape[1], LATO, LATO)


class TestaConv(nn.Module):
    def __init__(self, canali, n_scalari, k=32, h=64):
        super().__init__()
        strati = [nn.Conv2d(canali, k, 3), nn.ReLU()]
        for _ in range(LATO // 2 - 1):                       # 5: 5->3->1; 7: 7->5->3->1
            strati += [nn.Conv2d(k, k, 3), nn.ReLU()]
        self.conv = nn.Sequential(*strati)
        self.mlp = nn.Sequential(nn.Linear(k + n_scalari, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, finestra, scalari):
        return self.mlp(torch.cat([self.conv(finestra).flatten(1), scalari], dim=1))


def fold(cubo, pesi, tr, va, taglia_zero, seed=SEED):
    torch.manual_seed(seed)
    m = tr[SCALARI].values.mean(axis=0)
    s = tr[SCALARI].values.std(axis=0)
    s[s == 0] = 1
    def tensori(df):
        return (torch.tensor(df["giorno_i"].values), torch.tensor(df["staz_i"].values),
                torch.tensor((df[SCALARI].values - m) / s, dtype=torch.float32))
    g, st, sc = tensori(tr)
    y = torch.tensor(tr["residual"].values, dtype=torch.float32).unsqueeze(1)
    gv, stv, scv = tensori(va)
    av, ov = va["aurora"].values, va["obs"].values
    net = TestaConv(cubo.shape[1], len(SCALARI))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    out = []
    for ep in range(1, MAX_EPOCHS + 1):
        net.train()
        perm = torch.randperm(len(y))
        for i in range(0, len(y), BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            pred = net(finestre(cubo, g[b], st[b], *pesi), sc[b])
            torch.nn.functional.mse_loss(pred, y[b]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            pv = torch.cat([net(finestre(cubo, gv[i:i + BATCH], stv[i:i + BATCH], *pesi), scv[i:i + BATCH])
                            for i in range(0, len(gv), BATCH)]).numpy().ravel()
        corr = av + pv
        if taglia_zero:
            corr = np.clip(corr, 0, None)
        e = corr - ov
        out.append({"epoca": ep, "rmse": float(np.sqrt(np.mean(e ** 2))), "bias": float(e.mean()),
                    "rmse_aurora": float(np.sqrt(np.mean((av - ov) ** 2))), "n": len(va)})
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
    print(f"campi: {tuple(cubo.shape)} (giorni, canali, lat, lon); canali: {nomi}", flush=True)
    st = t.drop_duplicates("uid")[["uid", "lat", "lon"]].reset_index(drop=True)
    pesi = pesi_finestra(st["lat"].values, st["lon"].values, glat, glon)
    t["staz_i"] = t["uid"].map(pd.Series(st.index, index=st["uid"]))
    t["giorno_i"] = giorni.get_indexer(t["date"])
    t = t[t["giorno_i"] >= 0]
    print(f"righe: {len(t)}, stazioni: {len(st)}", flush=True)

    rows = []
    for param in PARAMS:
        tp = t[t["parameter"] == param]
        for reg in REGIONS:
            for anno in anni:
                tr = tp[(tp["regione"] != reg) & (tp["anno"] != anno)]
                va = tp[(tp["regione"] == reg) & (tp["anno"] == anno)]
                if tr.empty or va.empty:
                    continue
                for r in fold(cubo, pesi, tr, va, param in NON_NEGATIVI):
                    rows.append({"parametro": param, "regione": reg, "anno": anno, **r})
                best = min(x["rmse"] for x in rows[-MAX_EPOCHS:])
                print(f"[{param}] {reg} {anno}: migliore {best:.3f} (Aurora {rows[-1]['rmse_aurora']:.3f})", flush=True)
                pd.DataFrame(rows).to_csv(RISULTATI, index=False)

    res = pd.DataFrame(rows)
    curva = res.groupby(["parametro", "epoca"])["rmse"].mean().unstack("parametro")
    conv = pd.DataFrame({"rmse_conv": curva.min(), "epoche_conv": curva.idxmin()})
    # testa attuale (MLP, 28 feature nel punto): stessa CV, risultati di cv_regione_anno.py
    mlp = pd.read_csv("risultati/cv_regione_anno_risultati_modello_precedente.csv")  # testa con cui e' stato fatto il confronto
    cm = mlp.groupby(["parametro", "epoca"])["rmse_corretto"].mean().unstack("parametro")
    conv["rmse_testa_attuale"] = cm.min()
    conv["rmse_aurora"] = res.groupby("parametro")["rmse_aurora"].mean()
    conv["differenza_%"] = 100 * (conv["rmse_conv"] - conv["rmse_testa_attuale"]) / conv["rmse_testa_attuale"]
    print("\n=== CV regione x anno: convoluzionale vs testa attuale (RMSE medio sui fold) ===")
    print(conv.round(3))
    sel = pd.concat([res[(res["parametro"] == p) & (res["epoca"] == e)] for p, e in conv["epoche_conv"].items()])
    best_mlp = pd.concat([mlp[(mlp["parametro"] == p) & (mlp["epoca"] == e)] for p, e in cm.idxmin().items()])
    cmp_ = sel.set_index(["parametro", "regione", "anno"])["rmse"].to_frame("conv").join(
        best_mlp.set_index(["parametro", "regione", "anno"])["rmse_corretto"].rename("testa_attuale"))
    print("\nper fold:")
    print(cmp_.round(2))
    print("\nfold in cui la convoluzionale e' migliore:",
          cmp_.assign(m=cmp_["conv"] < cmp_["testa_attuale"]).groupby("parametro")["m"].sum().to_dict())


if __name__ == "__main__":
    main()
