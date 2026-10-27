"""
Salva i CAMPI giornalieri di Aurora ritagliati sull'AOI (con un margine), per pochi giorni scelti: 
servono per fare le mappe ad alta risoluzione applicando le teste in ogni pixel (mappa_downscaling.py).

Stessa definizione del giorno usata per il training: condizioni iniziali ERA5 (ARCO) alle 18 UTC
del giorno prima e alle 00 UTC del giorno D, 4 passi da 6 h con previsioni orarie -> 24 valori
orari (01-24 UTC); per ogni cella: massimo e minimo della temperatura a 2 m, somma della pioggia
oraria, media della radiazione oraria.

Valori GREZZI (nessuna conversione di unita'), come in aurora_daily_{2016,2022,2023}.csv con cui
sono state allenate le teste: il fattore di scala per rr7/qq2 lo applica mappa_downscaling.py,
ricavato dalla tabella di training. (La versione attuale di extract_aurora_daily_arco.py converte
pioggia x1000 e radiazione /3600, ma i file di training sono stati prodotti senza conversione.)

Salva TUTTE le variabili che Aurora produce (il costo di GPU e' lo stesso): le 4 usate oggi dal
training (stessi nomi e valori grezzi), la media giornaliera di tutte le variabili di superficie
(sup_<nome>_media) e di quelle in quota su tutti i 13 livelli (atm_<nome>_media, dimensione
"livello" in hPa), il massimo orario del vento a 10 m e le variabili statiche (orografia in m,
maschera terra/mare, tipo di suolo): servono per una testa con piu' variabili o per una rete
convoluzionale sulla finestra intorno al punto.

Prerequisiti (Lightning, GPU):
    pip install torch microsoft-aurora xarray gcsfs zarr huggingface_hub pandas numpy netcdf4

Uso:
    python extract_aurora_campi.py 2021-08-15 2021-10-04 2021-12-31 2016-06-15   # giorni scelti
    python extract_aurora_campi.py 2016 2022 2023 2020                            # anni interi
    python extract_aurora_campi.py 2020:3                    # un giorno ogni 3 dal 1 gennaio 2020
    python extract_aurora_campi.py --poca-memoria 2016-06-15  # per GPU piccole (T4 di Colab gratuito)
    (AuroraV1p5 calcola gia' di default in float16 con torch.autocast, tenendo pero' i pesi in float32
    e una loro copia in float16. --poca-memoria tiene i pesi degli strati lineari direttamente in
    float16: e' la stessa conversione che l'autocast fa prima di ogni calcolo, quindi i risultati non
    dovrebbero cambiare. Inoltre il decoder dei livelli di pressione, che tratta ogni punto della
    griglia in modo indipendente, viene eseguito a pezzi (stesso risultato, picco di memoria ~8 volte
    piu' basso). Va comunque verificato su un giorno di training, vedi colab_aurora_campi.ipynb)
    (per allenare le teste con Aurora interpolata bilinearmente servono gli anni interi di
    training, 2016 2022 2023, e l'anno di test 2020; ~ le stesse ore delle estrazioni precedenti.
    Resumable: i giorni gia' salvati vengono saltati, si puo' interrompere e rilanciare.)
    (2016-06-15 e' un giorno di training: serve a mappa_downscaling.py per controllare che i campi
    nella cella piu' vicina coincidano con aurora_daily_2016.csv, cioe' che l'input sia lo stesso
    con cui sono state allenate le teste)
Output:
    aurora_campi/aurora_campi_<giorno>.nc  (tx3_aurora, tn3_aurora, rr7_aurora_raw, qq2_aurora_raw,
                               orog_aurora, sup_*, atm_*, statica_*; dimensioni lat x lon,
                               livello x lat x lon per le variabili in quota)
"""
import os
import pickle
import sys
import time
from pathlib import Path

# riduce la frammentazione della memoria GPU (sulla T4 di Colab ~3 GB restavano riservati ma inutilizzati)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
import xarray as xr
from huggingface_hub import hf_hub_download

CARTELLA = Path("aurora_campi")


def _giorni(argomenti):
    """Argomenti: giorni (AAAA-MM-GG), anni interi (AAAA) e/o un giorno ogni N di un anno (AAAA:N)."""
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
# senza argomenti: anni interi di training (2016, 2022, 2023)
GIORNI = _giorni([a for a in sys.argv[1:] if a not in OPZIONI] or ["2016", "2022", "2023"])
# AOI 44.0-46.5 N, 6.5-13.0 E piu' un margine di 0.5 gradi (2 celle), cosi' l'interpolazione
# bilineare ha le celle vicine anche sul bordo. AOI + bordo
LAT_MIN, LAT_MAX, LON_MIN, LON_MAX = 43.5, 47.0, 6.0, 13.5

ARCO_STORE = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
FINE_LEAD_TIMES = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
STEPS = 4  # 4 x 6h = 24h, copre l'intero giorno D
SURF_MAP = {"2t": "2m_temperature", "10u": "10m_u_component_of_wind",
            "10v": "10m_v_component_of_wind", "msl": "mean_sea_level_pressure"}
ATMOS_MAP = {"t": "temperature", "u": "u_component_of_wind", "v": "v_component_of_wind",
             "q": "specific_humidity", "z": "geopotential"}


def load_static_vars():
    static_path = hf_hub_download(repo_id="microsoft/aurora", filename="aurora-0.25-v1.5-static.pickle")
    with open(static_path, "rb") as f:
        static_raw = pickle.load(f)
    return {k: torch.from_numpy(v) for k, v in static_raw.items()}


def get_time_index(ds, cache_path="arco_time_cache.npy"):
    p = Path(cache_path) # costruisce il path
    if p.exists():
        return np.load(p)
    print("prima volta: carico la coordinata tempo di ARCO-ERA5 (puo' volerci qualche minuto)...")
    tvals = ds["time"].values
    np.save(p, tvals) 
    return tvals


def idx_for(time_index, target_ts):
    target = np.datetime64(target_ts)
    i = np.searchsorted(time_index, target)
    if i >= len(time_index) or time_index[i] != target:
        raise KeyError(f"{target_ts} non trovato nella coordinata tempo di ARCO-ERA5")
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
    """Esegue modulo(latents, x) a pezzi lungo la prima dimensione (punti della griglia, indipendenti
    tra loro nel decoder dei livelli): stesso risultato, meno memoria di picco."""
    originale = modulo.forward

    def forward(latents, x):
        k = -(-latents.shape[0] // pezzi)
        return torch.cat([originale(latents[i:i + k], x[i:i + k]) for i in range(0, latents.shape[0], k)])

    modulo.forward = forward


def main():
    from aurora import AuroraV1p5, rollout

    print("apro ARCO-ERA5 (solo metadati, lazy)...")
    ds = xr.open_zarr(ARCO_STORE, chunks=None, storage_options=dict(token="anon"))
    level_idx = [int(np.where(ds.level.values == lv)[0][0]) for lv in PRESSURE_LEVELS]
    time_index = get_time_index(ds)

    static_vars = load_static_vars()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AuroraV1p5()
    model.load_checkpoint("microsoft/aurora", "aurora-0.25-v1.5.ckpt")
    # pesi non da allenare: altrimenti l'autocast tiene in memoria una copia in float16 di tutti i pesi
    model.requires_grad_(False)
    if POCA_MEMORIA:
        if not model.autocast:
            sys.exit("--poca-memoria vale solo se il modello usa l'autocast (AuroraV1p5 lo usa di default)")
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                m.to(model.autocast_dtype)
        for nome in ("level_decoder", "level_decoder_alternate"):
            if isinstance(getattr(model.decoder, nome, None), torch.nn.Module):
                a_pezzi(getattr(model.decoder, nome))
    model.eval().to(device)
    print(f"modello su {device}, autocast {model.autocast_dtype if model.autocast else 'no'}"
          + (", pesi degli strati lineari in " + str(model.autocast_dtype) + ", decoder dei livelli a pezzi"
             if POCA_MEMORIA else ""))

    CARTELLA.mkdir(exist_ok=True)
    fatti = 0
    for giorno in GIORNI:
        day = pd.Timestamp(giorno)
        out = CARTELLA / f"aurora_campi_{day.date()}.nc"
        if out.exists():  # resumable: i giorni gia' fatti si saltano
            continue
        t0 = time.time()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()  # picco di memoria del solo giorno corrente
        batch = make_batch(ds, static_vars, idx_for(time_index, day - pd.Timedelta(hours=6)),
                           idx_for(time_index, day), level_idx)
        orari_sup = {}   # variabile di superficie -> lista di campi orari (lat x lon)
        orari_atm = {}   # variabile in quota -> lista di campi orari (livello x lat x lon)
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
                    # ultime due dimensioni = lat, lon
                    return x.index_select(-2, iy_t).index_select(-1, ix_t).float().cpu().numpy()

                for k, v in pred.surf_vars.items():          # tutte le variabili di superficie
                    orari_sup.setdefault(k, []).append(ritaglia(v[0, 0]))
                for k, v in pred.atmos_vars.items():         # tutte le variabili in quota, tutti i livelli
                    orari_atm.setdefault(k, []).append(ritaglia(v[0, 0]))
        orari_sup = {k: np.array(v) for k, v in orari_sup.items()}   # ora x lat x lon
        orari_atm = {k: np.array(v) for k, v in orari_atm.items()}   # ora x livello x lat x lon
        n_ore = len(orari_sup["2t"])
        if n_ore != 24:
            print(f"ATTENZIONE {giorno}: {n_ore} valori orari invece di 24")

        # Variabili usate dal training attuale: stessi nomi e stessi valori grezzi di aurora_daily_*.csv
        t2 = orari_sup["2t"] - 273.15
        dati = {"tx3_aurora": (("lat", "lon"), t2.max(axis=0)),
                "tn3_aurora": (("lat", "lon"), t2.min(axis=0)),
                "rr7_aurora_raw": (("lat", "lon"), orari_sup["scaled_tp_1h"].sum(axis=0)),
                "qq2_aurora_raw": (("lat", "lon"), orari_sup["ssrd_1h"].mean(axis=0))}
        # Tutte le variabili di superficie: media giornaliera (unita' native di Aurora)
        for k, v in orari_sup.items():
            dati[f"sup_{k}_media"] = (("lat", "lon"), v.mean(axis=0))
        # extra utile: massimo orario della velocita' del vento a 10 m
        # (l'escursione termica non si salva: e' tx3_aurora - tn3_aurora)
        if "10u" in orari_sup and "10v" in orari_sup:
            dati["sup_vento10_max"] = (("lat", "lon"), np.hypot(orari_sup["10u"], orari_sup["10v"]).max(axis=0))
        # Tutte le variabili in quota su tutti i livelli: media giornaliera
        for k, v in orari_atm.items():
            dati[f"atm_{k}_media"] = (("livello", "lat", "lon"), v.mean(axis=0))

        # Variabili statiche (orografia = geopotenziale / g, maschera terra/mare, tipo di suolo)
        lat_static = np.linspace(90, -90, static_vars["z"].shape[0])
        righe = [int(np.abs(lat_static - la).argmin()) for la in lat[iy]]
        for k, v in static_vars.items():
            arr = v.cpu().numpy()[np.ix_(righe, ix)]
            dati["orog_aurora" if k == "z" else f"statica_{k}"] = (("lat", "lon"), arr / 9.80665 if k == "z" else arr)

        campi = xr.Dataset(
            dati,
            coords={"lat": lat[iy], "lon": lon[ix], "livello": livelli},
            attrs={"giorno": str(day.date()), "ore": n_ore,
                   "nota": "tx3/tn3/rr7_raw/qq2_raw come aurora_daily_*.csv (grezzi); sup_*/atm_* = medie "
                           "giornaliere delle 24 previsioni orarie, unita' native di Aurora; livello in hPa"})
        campi.to_netcdf(out)
        fatti += 1
        memoria = f", memoria GPU max {torch.cuda.max_memory_allocated() / 1e9:.1f} GB" if device == "cuda" else ""
        print(f"salvato {out} ({fatti} nuovi, {len(GIORNI)} richiesti) in {(time.time() - t0) / 60:.1f} min"
              f"{memoria}", flush=True)
        del batch, pred
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
