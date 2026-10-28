"""
Saves the daily Aurora FIELDS cropped to the AOI (with a margin), for the chosen days: they are
needed to build the high-resolution maps by applying the heads in each pixel (mappa_downscaling.py).

Same definition of the day as in training: ERA5 initial conditions (ARCO) at 18 UTC of the previous
day and at 00 UTC of day D, 4 steps of 6 h with hourly forecasts -> 24 hourly values (01-24 UTC);
for each cell: maximum and minimum of the 2 m temperature, sum of the hourly precipitation, mean of
the hourly radiation.

RAW values (no unit conversion), as in aurora_daily_{2016,2022,2023}.csv with which the heads were
trained: the scale factor for rr7/qq2 is applied by mappa_downscaling.py, derived from the training
table. (The current version of extract_aurora_daily_arco.py converts precipitation x1000 and
radiation /3600, but the training files were produced without conversion.)

Saves ALL the variables Aurora produces (the GPU cost is the same): the 4 used by the current
training (same names and raw values), the daily mean of all the surface variables
(sup_<name>_media) and of the pressure-level ones on all 13 levels (atm_<name>_media, dimension
"livello" in hPa), the hourly maximum of the 10 m wind and the static variables (orography in m,
land/sea mask, soil type): they are needed for a head with more variables or for a convolutional
network on the window around the point.

Note: only 4 of the 19 surface inputs that AuroraV1p5 expects are given (SURF_MAP below); the
Aurora code accepts the subset without errors. This gives a strong cold bias in summer, corrected by
the downscaling. The inputs are kept unchanged so that training, test and maps are consistent.

Requirements (Lightning, GPU):
    pip install torch microsoft-aurora xarray gcsfs zarr huggingface_hub pandas numpy netcdf4

Usage:
    python extract_aurora_campi.py 2021-08-15 2021-10-04 2021-12-31 2016-06-15   # chosen days
    python extract_aurora_campi.py 2016 2022 2023 2020                            # whole years
    python extract_aurora_campi.py 2020:3                    # one day every 3 from 1 January 2020
    python extract_aurora_campi.py --poca-memoria 2016-06-15  # for small GPUs (free Colab T4)
    (AuroraV1p5 already computes in float16 by default with torch.autocast, keeping however the
    weights in float32 plus a float16 copy of them. --poca-memoria keeps the weights of the linear
    layers directly in float16: it is the same conversion autocast does before each computation, so
    the results should not change. In addition the pressure-level decoder, which treats each grid
    point independently, is run in chunks (same result, peak memory ~8 times lower). It should still
    be checked on a training day, see colab_aurora_campi.ipynb)
    (to train the heads with bilinearly interpolated Aurora the whole training years 2016 2022 2023
    and the test year 2020 are needed; ~ the same hours as the previous extractions.
    Resumable: days already saved are skipped, it can be interrupted and restarted.)
    (2016-06-15 is a training day: mappa_downscaling.py uses it to check that the fields in the
    nearest cell match aurora_daily_2016.csv, i.e. that the input is the same as the one the heads
    were trained with)
Output:
    aurora_campi/aurora_campi_<day>.nc  (tx3_aurora, tn3_aurora, rr7_aurora_raw, qq2_aurora_raw,
                               orog_aurora, sup_*, atm_*, statica_*; dimensions lat x lon,
                               livello x lat x lon for the pressure-level variables)
"""
import os
import pickle
import sys
import time
from pathlib import Path

# reduces GPU memory fragmentation (on the Colab T4 ~3 GB stayed reserved but unused)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
import xarray as xr
from huggingface_hub import hf_hub_download

# Paths relative to the main project folder (where the script is run from)
CACHE_TEMPI_ARCO = "dati/arco_time_cache.npy"
CARTELLA = Path("dati/aurora_campi")


def _giorni(argomenti):
    """Arguments: days (YYYY-MM-DD), whole years (YYYY) and/or one day every N of a year (YYYY:N)."""
    giorni = []
    for a in argomenti:
        if ":" in a:
            anno, passo = a.split(":")
            giorni += [str(d.date()) for d in pd.date_range(f"{anno}-01-01", f"{anno}-12-31", freq=f"{int(passo)}D")]
        elif len(a) == 4 and a.isdigit():
            giorni += [str(d.date()) for d in pd.date_range(f"{a}-01-01", f"{a}-12-31", freq="D")]
        else:
            giorni.append(a)
    return giorni


OPZIONI = {"--poca-memoria"}
POCA_MEMORIA = "--poca-memoria" in sys.argv
# without arguments: whole training years (2016, 2022, 2023)
GIORNI = _giorni([a for a in sys.argv[1:] if a not in OPZIONI] or ["2016", "2022", "2023"])
# AOI 44.0-46.5 N, 6.5-13.0 E plus a margin of 0.5 degrees (2 cells), so that the bilinear
# interpolation has the neighbouring cells on the border too. AOI + border
LAT_MIN, LAT_MAX, LON_MIN, LON_MAX = 43.5, 47.0, 6.0, 13.5

ARCO_STORE = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
FINE_LEAD_TIMES = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
STEPS = 4  # 4 x 6h = 24h, covers the whole day D
SURF_MAP = {"2t": "2m_temperature", "10u": "10m_u_component_of_wind",
            "10v": "10m_v_component_of_wind", "msl": "mean_sea_level_pressure"}
ATMOS_MAP = {"t": "temperature", "u": "u_component_of_wind", "v": "v_component_of_wind",
             "q": "specific_humidity", "z": "geopotential"}


def load_static_vars():
    static_path = hf_hub_download(repo_id="microsoft/aurora", filename="aurora-0.25-v1.5-static.pickle")
    with open(static_path, "rb") as f:
        static_raw = pickle.load(f)
    return {k: torch.from_numpy(v) for k, v in static_raw.items()}


def get_time_index(ds, cache_path=CACHE_TEMPI_ARCO):
    p = Path(cache_path) # builds the path
    if p.exists():
        return np.load(p)
    print("first time: loading the ARCO-ERA5 time coordinate (it may take a few minutes)...")
    tvals = ds["time"].values
    np.save(p, tvals)
    return tvals


def idx_for(time_index, target_ts):
    target = np.datetime64(target_ts)
    i = np.searchsorted(time_index, target)
    if i >= len(time_index) or time_index[i] != target:
        raise KeyError(f"{target_ts} not found in the ARCO-ERA5 time coordinate")
    return int(i)


def make_batch(ds, static_vars, t_idx0, t_idx1, levels_idx):
    from aurora import Batch, Metadata

    def surf(short):
        return torch.from_numpy(ds[SURF_MAP[short]].isel(time=[t_idx0, t_idx1]).values[None].astype("float32"))

    def atmos(short):
        return torch.from_numpy(
            ds[ATMOS_MAP[short]].isel(time=[t_idx0, t_idx1], level=levels_idx).values[None].astype("float32"))

    return Batch(
        surf_vars={k: surf(k) for k in SURF_MAP},
        static_vars=static_vars,
        atmos_vars={k: atmos(k) for k in ATMOS_MAP},
        metadata=Metadata(
            lat=torch.from_numpy(ds.latitude.values.astype("float64")),
            lon=torch.from_numpy(ds.longitude.values.astype("float64")),
            time=(pd.Timestamp(ds.time.isel(time=t_idx1).values).to_pydatetime(),),
            atmos_levels=tuple(PRESSURE_LEVELS),
        ),
    )


def a_pezzi(modulo, pezzi=8):
    """Runs modulo(latents, x) in chunks along the first dimension (grid points, independent of
    each other in the level decoder): same result, lower peak memory."""
    originale = modulo.forward

    def forward(latents, x):
        k = -(-latents.shape[0] // pezzi)
        return torch.cat([originale(latents[i:i + k], x[i:i + k]) for i in range(0, latents.shape[0], k)])

    modulo.forward = forward


def main():
    from aurora import AuroraV1p5, rollout

    print("opening ARCO-ERA5 (metadata only, lazy)...")
    ds = xr.open_zarr(ARCO_STORE, chunks=None, storage_options=dict(token="anon"))
    level_idx = [int(np.where(ds.level.values == lv)[0][0]) for lv in PRESSURE_LEVELS]
    time_index = get_time_index(ds)

    static_vars = load_static_vars()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AuroraV1p5()
    model.load_checkpoint("microsoft/aurora", "aurora-0.25-v1.5.ckpt")
    # weights are not trained: otherwise autocast keeps a float16 copy of all the weights in memory
    model.requires_grad_(False)
    if POCA_MEMORIA:
        if not model.autocast:
            sys.exit("--poca-memoria only applies if the model uses autocast (AuroraV1p5 does by default)")
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                m.to(model.autocast_dtype)
        for nome in ("level_decoder", "level_decoder_alternate"):
            if isinstance(getattr(model.decoder, nome, None), torch.nn.Module):
                a_pezzi(getattr(model.decoder, nome))
    model.eval().to(device)
    print(f"model on {device}, autocast {model.autocast_dtype if model.autocast else 'no'}"
          + (", linear layer weights in " + str(model.autocast_dtype) + ", level decoder in chunks"
             if POCA_MEMORIA else ""))

    CARTELLA.mkdir(parents=True, exist_ok=True)
    fatti = 0
    for giorno in GIORNI:
        day = pd.Timestamp(giorno)
        out = CARTELLA / f"aurora_campi_{day.date()}.nc"
        if out.exists():  # resumable: days already done are skipped
            continue
        t0 = time.time()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()  # peak memory of the current day only
        batch = make_batch(ds, static_vars, idx_for(time_index, day - pd.Timedelta(hours=6)),
                           idx_for(time_index, day), level_idx)
        orari_sup = {}   # surface variable -> list of hourly fields (lat x lon)
        orari_atm = {}   # pressure-level variable -> list of hourly fields (level x lat x lon)
        iy = ix = None
        with torch.inference_mode():
            for pred in rollout(model, batch, steps=STEPS, fine_lead_times=FINE_LEAD_TIMES):
                if iy is None:
                    lat = pred.metadata.lat.cpu().numpy()
                    lon = pred.metadata.lon.cpu().numpy()
                    livelli = [int(l) for l in pred.metadata.atmos_levels]
                    iy = np.where((lat >= LAT_MIN) & (lat <= LAT_MAX))[0]
                    ix = np.where((lon >= LON_MIN) & (lon <= LON_MAX))[0]
                    iy_t = torch.as_tensor(iy, device=pred.surf_vars["2t"].device)
                    ix_t = torch.as_tensor(ix, device=pred.surf_vars["2t"].device)

                def ritaglia(x):
                    # last two dimensions = lat, lon
                    return x.index_select(-2, iy_t).index_select(-1, ix_t).float().cpu().numpy()

                for k, v in pred.surf_vars.items():          # all the surface variables
                    orari_sup.setdefault(k, []).append(ritaglia(v[0, 0]))
                for k, v in pred.atmos_vars.items():         # all the pressure-level variables, all levels
                    orari_atm.setdefault(k, []).append(ritaglia(v[0, 0]))
        orari_sup = {k: np.array(v) for k, v in orari_sup.items()}   # hour x lat x lon
        orari_atm = {k: np.array(v) for k, v in orari_atm.items()}   # hour x level x lat x lon
        n_ore = len(orari_sup["2t"])
        if n_ore != 24:
            print(f"WARNING {giorno}: {n_ore} hourly values instead of 24")

        # Variables used by the current training: same names and same raw values as aurora_daily_*.csv
        t2 = orari_sup["2t"] - 273.15
        dati = {"tx3_aurora": (("lat", "lon"), t2.max(axis=0)),
                "tn3_aurora": (("lat", "lon"), t2.min(axis=0)),
                "rr7_aurora_raw": (("lat", "lon"), orari_sup["scaled_tp_1h"].sum(axis=0)),
                "qq2_aurora_raw": (("lat", "lon"), orari_sup["ssrd_1h"].mean(axis=0))}
        # All the surface variables: daily mean (native Aurora units)
        for k, v in orari_sup.items():
            dati[f"sup_{k}_media"] = (("lat", "lon"), v.mean(axis=0))
        # useful extra: hourly maximum of the 10 m wind speed
        # (the diurnal temperature range is not saved: it is tx3_aurora - tn3_aurora)
        if "10u" in orari_sup and "10v" in orari_sup:
            dati["sup_vento10_max"] = (("lat", "lon"), np.hypot(orari_sup["10u"], orari_sup["10v"]).max(axis=0))
        # All the pressure-level variables on all levels: daily mean
        for k, v in orari_atm.items():
            dati[f"atm_{k}_media"] = (("livello", "lat", "lon"), v.mean(axis=0))

        # Static variables (orography = geopotential / g, land/sea mask, soil type)
        lat_static = np.linspace(90, -90, static_vars["z"].shape[0])
        righe = [int(np.abs(lat_static - la).argmin()) for la in lat[iy]]
        for k, v in static_vars.items():
            arr = v.cpu().numpy()[np.ix_(righe, ix)]
            dati["orog_aurora" if k == "z" else f"statica_{k}"] = (("lat", "lon"), arr / 9.80665 if k == "z" else arr)

        campi = xr.Dataset(
            dati,
            coords={"lat": lat[iy], "lon": lon[ix], "livello": livelli},
            attrs={"giorno": str(day.date()), "ore": n_ore,
                   "nota": "tx3/tn3/rr7_raw/qq2_raw as in aurora_daily_*.csv (raw); sup_*/atm_* = daily means "
                           "of the 24 hourly forecasts, native Aurora units; livello in hPa"})
        campi.to_netcdf(out)
        fatti += 1
        memoria = f", max GPU memory {torch.cuda.max_memory_allocated() / 1e9:.1f} GB" if device == "cuda" else ""
        print(f"saved {out} ({fatti} new, {len(GIORNI)} requested) in {(time.time() - t0) / 60:.1f} min"
              f"{memoria}", flush=True)
        del batch, pred
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
