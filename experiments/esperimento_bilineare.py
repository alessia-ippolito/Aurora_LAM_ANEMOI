"""
Experiment: does the head improve if Aurora is bilinearly interpolated to the station location, and
even more if it also receives the other Aurora variables? Same region x year cross-validation as
cv_regione_anno.py (trains on 2 regions x 2 years, evaluates on the excluded region and year), same
exclusions (notebook §7.2), Piedmont never used.

No variant uses latitude and longitude: with lat/lon the network learns corrections tied to the
station coordinates and, in a held-out region, extrapolates them unstably (same data, only a
different row order: tx3 Veneto 2016 from 2.9 to 4.5 degrees; see also ablation_latlon_all.csv).
Two seeds per variant, to measure how stable the results are.

Variants (same network and same training, only the inputs change; labels kept in Italian, they are
the keys stored in the results file):
  A  cella vicina (nearest cell): aurora in the nearest cell ("aurora" column of the table) + elevation, season
  B  bilineare (bilinear):   aurora bilinearly interpolated at the exact station location + elevation, season
  C  bilineare+:   B + the other Aurora variables at the point (the 4 main variables together,
                   temperature and humidity aloft, boundary layer, cloudiness (ttr), wind, pressure,
                   snow, geopotential at 500 hPa) + difference station elevation - Aurora elevation
The target is always obs - aurora of the variant (for B and C the bilinear aurora). The error is
measured after each epoch: for each variant the best number of epochs on average over the folds is chosen.

Requires the Aurora fields of the training years in aurora_campi/ (extract_aurora_campi.py).

Usage:
    python esperimento_bilineare.py
Output:
    esperimento_bilineare_risultati.csv, summary on screen
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from campi_aurora import carica_campi, interpola_punti
from cv_comune import BATCH, PARAMS, REGIONS, SEED, assign_regions
from train_downscaling_head import DownscalingHead, applica_esclusioni

TABLE = "inutili/training_table_aurora_backup_cella_vicina.csv.gz"  # table with Aurora in the nearest cell (variant A)
MAX_EPOCHS = 20
SEEDS = [0, 1]
COL_RAW = {"tx3": "tx3_aurora", "tn3": "tn3_aurora", "rr7": "rr7_aurora_raw", "qq2": "qq2_aurora_raw"}
QC = {"tx3": (-30, 50), "tn3": (-40, 35), "qq2": (0, 450), "rr7": (0, 300)}  # clip of Aurora, as in §6.6
NON_NEGATIVI = {"rr7", "qq2"}
BASE = ["elev", "doy_sin", "doy_cos"]  # no lat/lon
EXTRA = ["a_tx3", "a_tn3", "a_rr7", "a_qq2",
         "sup_2t_media", "sup_msl_media", "sup_10u_media", "sup_10v_media", "sup_vento10_max",
         "sup_i10fg_media", "sup_blh_media", "sup_ttr_1h_media", "sup_scaled_sf_1h_media",
         "atm_t_925", "atm_t_850", "atm_t_700", "atm_t_500", "atm_q_925", "atm_q_850", "atm_q_700",
         "atm_z_500", "atm_u_850", "atm_v_850", "dz"]


def prepara(idw=False):
    """idw=True: adds a_rr7_idw, a_qq2_idw and aurora_D (like aurora_B, but precipitation and
    radiation interpolated with IDW instead of bilinearly; see esperimento_idw.py)."""
    t = pd.read_csv(TABLE, parse_dates=["date"])
    t = applica_esclusioni(t[t["split"] == "train"])
    anni = sorted(t["date"].dt.year.unique())
    st = t.drop_duplicates("uid")[["uid", "lat", "lon"]].reset_index(drop=True)
    campi = carica_campi(anni)
    print(f"Aurora fields: {dict(campi.sizes)}", flush=True)
    vic = interpola_punti(campi[list(COL_RAW.values())], st["lat"], st["lon"], "nearest", ids=st["uid"])
    bil = interpola_punti(campi, st["lat"], st["lon"], "linear", ids=st["uid"])
    vic = vic.rename(columns={"punto": "uid"})
    bil = bil.rename(columns={"punto": "uid"})

    # raw -> physical units factor as in the table (ratio aurora / raw in the nearest cell)
    fatt = {"tx3": 1.0, "tn3": 1.0}
    for p in NON_NEGATIVI:
        q = t[t["parameter"] == p].merge(vic[["uid", "date", COL_RAW[p]]], on=["uid", "date"])
        lo, hi = QC[p]
        q = q[(q["aurora"] > lo) & (q["aurora"] < hi) & (q[COL_RAW[p]] > 0)]
        fatt[p] = float((q["aurora"] / q[COL_RAW[p]]).median())
    print("scale factors:", fatt)
    for p, c in COL_RAW.items():
        bil[f"a_{p}"] = (bil[c] * fatt[p]).clip(*QC[p])
    servono = [c for c in EXTRA if c in bil.columns] + ["orog_aurora"]
    if idw:
        cols = [COL_RAW[p] for p in NON_NEGATIVI]
        ii = interpola_punti(campi[cols], st["lat"], st["lon"], "idw", ids=st["uid"])
        ii = ii.rename(columns={"punto": "uid"})
        bil = bil.merge(ii.rename(columns={COL_RAW[p]: f"raw_{p}_idw" for p in NON_NEGATIVI}), on=["uid", "date"])
        for p in NON_NEGATIVI:
            bil[f"a_{p}_idw"] = (bil[f"raw_{p}_idw"] * fatt[p]).clip(*QC[p])
            servono.append(f"a_{p}_idw")
    # only the needed columns, in float32 (all 119 columns on 2.2 million rows do not fit in memory)
    bil = bil[["uid", "date"] + servono].astype({c: "float32" for c in servono})

    t = t.merge(bil, on=["uid", "date"], how="inner")
    t["dz"] = t["elev"] - t["orog_aurora"]
    t["aurora_A"] = t["aurora"]
    t["aurora_B"] = np.select([t["parameter"] == p for p in PARAMS], [t[f"a_{p}"] for p in PARAMS])
    if idw:
        t["aurora_D"] = np.select([t["parameter"] == p for p in PARAMS],
                                  [t[f"a_{p}_idw" if p in NON_NEGATIVI else f"a_{p}"] for p in PARAMS])
    doy = t["date"].dt.dayofyear.astype(float)
    t["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    t["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    t["anno"] = t["date"].dt.year
    t = assign_regions(t)
    print(f"rows: {len(t)} | mean difference |aurora A - B|: "
          f"{t.groupby('parameter').apply(lambda g: (g.aurora_A - g.aurora_B).abs().mean(), include_groups=False).round(3).to_dict()}",
          flush=True)
    return t


VARIANTI = {"A cella vicina": ("aurora_A", BASE), "B bilineare": ("aurora_B", BASE),
            "C bilineare+": ("aurora_B", BASE + EXTRA)}


def fold(tr, va, col_aurora, feature, taglia_zero, seed=SEED):
    torch.manual_seed(seed)
    cols = [col_aurora] + feature
    mean = tr[cols].values.mean(axis=0)
    std = tr[cols].values.std(axis=0)
    std[std == 0] = 1.0
    x = torch.tensor((tr[cols].values - mean) / std, dtype=torch.float32)
    y = torch.tensor((tr["obs"] - tr[col_aurora]).values, dtype=torch.float32).unsqueeze(1)
    xv = torch.tensor((va[cols].values - mean) / std, dtype=torch.float32)
    av = va[col_aurora].values
    ov = va["obs"].values
    net = DownscalingHead(n_features=len(cols)).net
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(x, y), batch_size=BATCH, shuffle=True)
    out = []
    for ep in range(1, MAX_EPOCHS + 1):
        net.train()
        for xb, yb in loader:
            opt.zero_grad()
            torch.nn.functional.mse_loss(net(xb), yb).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            corr = av + net(xv).numpy().ravel()
        if taglia_zero:
            corr = np.clip(corr, 0, None)
        e = corr - ov
        out.append({"epoca": ep, "rmse": float(np.sqrt(np.mean(e ** 2))), "bias": float(e.mean()),
                    "rmse_aurora": float(np.sqrt(np.mean((av - ov) ** 2)))})
    return out


def main(varianti=VARIANTI, uscita="risultati/esperimento_bilineare_risultati.csv", idw=False):
    torch.set_num_threads(4)
    t = prepara(idw)
    servono = list(dict.fromkeys(c for col, feat in varianti.values() for c in [col] + feat))
    rows = []
    for param in PARAMS:
        tp = t[t["parameter"] == param].dropna(subset=["obs", "aurora_A", "aurora_B"] + BASE + EXTRA + servono)
        for reg in REGIONS:
            for anno in sorted(tp["anno"].unique()):
                tr = tp[(tp["regione"] != reg) & (tp["anno"] != anno)]
                va = tp[(tp["regione"] == reg) & (tp["anno"] == anno)]
                if tr.empty or va.empty:
                    continue
                for nome, (col, feat) in varianti.items():
                    for seed in SEEDS:
                        for r in fold(tr, va, col, feat, param in NON_NEGATIVI, seed=seed):
                            rows.append({"parametro": param, "regione": reg, "anno": anno, "variante": nome,
                                         "seed": seed, **r})
                print(f"[{param}] {reg} {anno}: done", flush=True)
        res = pd.DataFrame(rows)
        res.to_csv(uscita, index=False)

    res = pd.DataFrame(rows)
    curva = res.groupby(["parametro", "variante", "epoca"])["rmse"].mean()
    best = curva.groupby(["parametro", "variante"]).agg(["min", "idxmin"])
    best["epoca"] = best["idxmin"].apply(lambda x: x[2])
    aur = res.groupby(["parametro", "variante"])["rmse_aurora"].mean()
    # stability: difference between the two seeds (mean over the folds) at the chosen epoch
    sel = res.merge(best["epoca"].rename("ep_scelta").reset_index(), on=["parametro", "variante"])
    sel = sel[sel["epoca"] == sel["ep_scelta"]]
    per_seed = sel.groupby(["parametro", "variante", "seed"])["rmse"].mean().unstack("seed")
    out = pd.DataFrame({"rmse_aurora_input": aur, "rmse_testa": best["min"], "epoche": best["epoca"],
                        "diff_tra_seed": (per_seed.max(axis=1) - per_seed.min(axis=1))})
    print("\n=== Region x year CV, without lat/lon: mean RMSE over folds and seeds, at the best epoch ===")
    print(out.round(3))
    # how much the same fold changes between the two seeds (residual instability)
    f = sel.groupby(["parametro", "variante", "regione", "anno"])["rmse"].agg(lambda x: x.max() - x.min())
    print("\nmaximum difference between seeds on the same fold:")
    print(f.groupby(["parametro", "variante"]).max().round(3))


if __name__ == "__main__":
    main()
