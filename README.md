# Downscaling di Aurora sulla Pianura Padana

Correzione statistica (downscaling puntuale) delle previsioni giornaliere di
[Aurora](https://github.com/microsoft/aurora) (0.25°) con le osservazioni delle stazioni al suolo.
Una piccola rete per ciascuna variabile impara il residuo osservazione − Aurora. Aurora viene
interpolata bilinearmente nel punto della stazione, e la rete usa come feature le variabili di
Aurora in quel punto, la quota (DEM a 30 m), la stagione e, per tn3 e qq2, la forma del rilievo.

| Variabile | Significato |
|---|---|
| `tx3` | temperatura massima giornaliera (°C) |
| `tn3` | temperatura minima giornaliera (°C) |
| `rr7` | precipitazione giornaliera (mm) |
| `qq2` | radiazione globale media giornaliera (W/m²) |

- **Training:** Lombardia, Emilia-Romagna, Veneto e le altre stazioni dell'area, anni 2016, 2022 e 2023.
- **Validazione:** cross-validation regione × anno.
- **Test finale:** Piemonte 2020. È una regione e un anno mai visti, e va usato una sola volta.

## Struttura

```
dati/
  osservazioni/    cache delle osservazioni (ECA&D, ARPA Lombardia, ARPA Piemonte, ARPAE) e anagrafiche
  aurora_campi/    campi giornalieri di Aurora, un file .nc per giorno (extract_aurora_campi.py)
  statici/         DEM Copernicus GLO-30, confini delle regioni, SARAH-3
tabelle/           tabelle di training e test, fattori di scala, esclusioni (notebook)
modello/           epoche scelte in cross-validation e reti del modello finale (checkpoints/)
risultati/         risultati di cross-validation, esperimenti e test (.csv)
log/               output degli script
mappe/             mappe ad alta risoluzione e mappa dell'errore per stazione
experiments/       esperimenti che hanno portato al modello finale
inutili/           script e dati superati (non su git)
```

Tutti i percorsi sono relativi alla cartella principale. Ogni script e il notebook definiscono i
propri percorsi in cima al file. Gli script vanno lanciati **dalla cartella principale**.

## Requisiti

Python 3.12 con i seguenti pacchetti:

```
pip install numpy pandas pyarrow xarray netCDF4 scipy torch geopandas shapely pyproj rasterio \
            matplotlib cartopy requests cdsapi
```

L'estrazione dei campi di Aurora richiede una GPU e pacchetti in più (vedi il passo 2).

Il DEM Copernicus GLO-30 (`cop30_padania.tif`) e i file giornalieri SARAH-3 (`SISdm*.nc`) vanno
messi a mano in `dati/statici/`, perché il codice attuale non li scarica. I confini delle regioni
invece li scarica il notebook la prima volta che lo esegui (da openpolis/geojson-italy).

## Pipeline

### 1. Osservazioni: `padania_cerra_ecad_analysis.ipynb` (§1–§6.5)

Scarica e mette in cache in `dati/osservazioni/`:

- le stazioni ECA&D (API OGC EDR);
- ARPA Lombardia, ARPA Piemonte e ARPAE;

e ne analizza la completezza. Le cache esistenti vengono riusate, quindi rieseguire il notebook non
riscarica i dati.

### 2. Campi di Aurora: `extract_aurora_campi.py` (GPU)

Fa girare Aurora 0.25° v1.5 da ERA5 (ARCO) e salva un file per giorno in `dati/aurora_campi/`, con
le quattro variabili e tutte le altre variabili di superficie e in quota, ritagliate sull'area.

```
pip install torch microsoft-aurora xarray gcsfs zarr huggingface_hub pandas numpy netcdf4
python extract_aurora_campi.py 2016 2022 2023 2020          # anni di training e di test
python extract_aurora_campi.py 2021-08-15 2021-10-04        # giorni singoli, per le mappe
```

Lo script è riprendibile: i giorni già salvati vengono saltati. È pensato per Lightning AI. Su
Colab c'è `colab_aurora_campi.ipynb` (opzione `--poca-memoria` per GPU piccole).

### 3. Tabelle di training e test: notebook §6.6–§7

Costruisce in `tabelle/`:

- `training_table_aurora.parquet` e `test_table_aurora.parquet`: osservazione, Aurora interpolata
  bilinearmente nel punto della stazione, feature, quota DEM e residuo;
- `fattori_scala_aurora.json`: unità di rr7/qq2 ed elenco delle feature.

Il controllo qualità (§7) scrive `esclusioni_training.csv`, cioè le righe da non usare. La tabella
resta intatta: le esclusioni le applicano gli script.

Sempre nel §6.6 si applicano le correzioni delle coordinate: UTM per ARPA Lombardia e
`CORREZIONI_MANUALI`.

### 4. Cross-validation e scelta delle epoche

```
python cv_regione_anno.py > log/cv_regione_anno.log
```

Ci sono 9 fold (3 regioni × 3 anni). Il Piemonte non entra mai. Lo script scrive:

- `modello/epoche_cv.json`, letto dal passo 5;
- `risultati/cv_regione_anno_risultati.csv`.

### 5. Modello finale

```
python train_downscaling_head.py                    # tutte e 4 le variabili (--parameter tx3 per una sola)
python train_downscaling_head.py --valuta_test      # UNA volta sola, a modello definitivo
```

Allena su tutta la tabella di training con le epoche del passo 4 e salva
`modello/checkpoints/<variabile>_finale.pt`. Ogni file contiene l'ensemble di reti, le feature e
la loro normalizzazione.

### 6. Mappe e valutazione

```
python mappa_downscaling.py 2021-08-15 2021-10-04 2021-12-31   # mappe ~1 km, servono i campi di quei giorni
python mappa_rmse.py                                            # errore per stazione sul test (Piemonte 2020)
```

Le mappe vanno in `mappe/` e i confronti con le stazioni in `risultati/`.

## Esperimenti (`experiments/`)

Ogni esperimento usa la stessa cross-validation regione × anno (`cv_comune.py`) e cambia una sola
cosa rispetto al modello di riferimento. I risultati vanno in `risultati/`. Gli esperimenti si
lanciano come moduli dalla cartella principale:

```
python -m experiments.esperimento_terreno > log/esperimento_terreno.log
```

| Script | Domanda |
|---|---|
| `esperimento_bilineare` | Aurora interpolata bilinearmente invece della cella più vicina? |
| `esperimento_idw` | IDW per pioggia e radiazione invece della bilineare? |
| `esperimento_terreno` | descrittori del terreno oltre alla quota? (richiede `python descrittori_terreno.py`) |
| `esperimento_pioggia` | trasformazioni del target per la pioggia (radice, log)? |
| `esperimento_iperparametri` | dimensione della rete, dropout, learning rate |
| `esperimento_ensemble` | media di più reti con seed diversi |
| `esperimento_feature_ridotte` | si possono togliere feature ridondanti? |
| `esperimento_importanza` | permutation importance del modello finale |
| `esperimento_convoluzionale`, `_reg` | rete convoluzionale sulla finestra di Aurora intorno al punto |
| `esperimento_sarah_c` | pre-training della radiazione sui pixel SARAH-3 |

`riferimenti_semplici.py` (nella cartella principale) confronta la rete con alcune correzioni
semplici: correzione di quota, media mensile, regressione lineare.

## Cosa non è nel repository

Dati, campi di Aurora, tabelle e DEM sono esclusi da `.gitignore` perché sono grandi o si possono
ricreare con i passi 1–3. Sono inclusi codice, notebook, risultati (`.csv`), log, mappe in `.png` e
le reti del modello finale (`modello/checkpoints/*_finale.pt`).
