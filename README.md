# Downscaling Aurora over the Po Valley

Statistical correction (point downscaling) of the daily forecasts of
[Aurora](https://github.com/microsoft/aurora) (0.25°) with ground station observations.
One small neural network per variable learns the residual observation − Aurora. Aurora is
bilinearly interpolated to the station location, and the network uses as features the Aurora
variables at that point, the elevation (30 m DEM), the season and, for tn3 and qq2, the shape of
the terrain.

| Variable | Meaning |
|---|---|
| `tx3` | daily maximum temperature (°C) |
| `tn3` | daily minimum temperature (°C) |
| `rr7` | daily precipitation (mm) |
| `qq2` | daily mean global radiation (W/m²) |

- **Training:** Lombardy, Emilia-Romagna, Veneto and the other stations in the area, years 2016, 2022 and 2023.
- **Validation:** region × year cross-validation.
- **Final test:** Piedmont 2020. It is a region and a year never seen in training, and it is used only once.

## Structure

```
dati/
  osservazioni/    observation caches (ECA&D, ARPA Lombardia, ARPA Piemonte, ARPAE) and station registries
  aurora_campi/    daily Aurora fields, one .nc file per day (extract_aurora_campi.py)
  statici/         Copernicus GLO-30 DEM, region boundaries, SARAH-3
tabelle/           training and test tables, scale factors, exclusions (notebook)
modello/           epochs chosen in cross-validation and networks of the final model (checkpoints/)
risultati/         cross-validation, experiment and test results (.csv)
log/               script output
mappe/             high-resolution maps and per-station error map
experiments/       experiments that led to the final model
inutili/           superseded scripts and data (not on git)
```

All paths are relative to the main folder. Each script and the notebook define their own paths at
the top of the file. Scripts must be run **from the main folder**.

File, folder, function and variable names are in Italian; comments, documentation and figures are
in English. A short glossary:

| Italian | English | Italian | English |
|---|---|---|---|
| dati | data | osservazioni | observations |
| tabelle | tables | campi | fields |
| modello | model | statici | static data |
| risultati | results | esclusioni | exclusions |
| mappe | maps | fattori di scala | scale factors |
| esperimento | experiment | descrittori del terreno | terrain descriptors |
| regione / anno | region / year | forma del rilievo | terrain shape |
| riferimenti semplici | simple baselines | inutili | unused |

## Requirements

Python 3.12 with the following packages:

```
pip install numpy pandas pyarrow xarray netCDF4 scipy torch geopandas shapely pyproj rasterio \
            matplotlib cartopy requests cdsapi
```

Extracting the Aurora fields needs a GPU and a few more packages (see step 2).

The Copernicus GLO-30 DEM (`cop30_padania.tif`) and the daily SARAH-3 files (`SISdm*.nc`) must be
placed by hand in `dati/statici/`, because the current code does not download them. The region
boundaries are downloaded by the notebook the first time it runs (from openpolis/geojson-italy).

## Pipeline

### 1. Observations: `data_preparation_train_test.ipynb` (§1–§6.5)

Downloads and caches in `dati/osservazioni/`:

- the ECA&D stations (OGC EDR API);
- ARPA Lombardia, ARPA Piemonte and ARPAE;

and analyses their completeness. Existing caches are reused, so re-running the notebook does not
download the data again.

### 2. Aurora fields: `extract_aurora_campi.py` (GPU)

Runs Aurora 0.25° v1.5 from ERA5 (ARCO) and saves one file per day in `dati/aurora_campi/`, with
the four variables and all the other surface and pressure-level variables, cropped to the area.

```
pip install torch microsoft-aurora xarray gcsfs zarr huggingface_hub pandas numpy netcdf4
python extract_aurora_campi.py 2016 2022 2023 2020          # training and test years
python extract_aurora_campi.py 2021-08-15 2021-10-04        # single days, for the maps
```

The script can be resumed: days already saved are skipped. It was written for Lightning AI. For
Colab there is `colab_aurora_campi.ipynb` (option `--poca-memoria` for small GPUs).

Note: Aurora is run with only 4 of the 19 surface inputs that AuroraV1p5 expects (2 m temperature,
10 m wind, mean sea level pressure). This gives Aurora a strong cold bias in summer, which the
downscaling largely corrects. "Aurora" in this project therefore means Aurora with reduced input.

### 3. Training and test tables: notebook §6.6–§7

Builds in `tabelle/`:

- `training_table_aurora.parquet` and `test_table_aurora.parquet`: observation, Aurora bilinearly
  interpolated to the station location, features, DEM elevation and residual;
- `fattori_scala_aurora.json`: units of rr7/qq2 and list of features.

The quality control (§7) writes `esclusioni_training.csv`, the rows not to be used. The tables stay
intact: the scripts apply the exclusions.

The coordinate corrections are also applied in §6.6: UTM for ARPA Lombardia and
`CORREZIONI_MANUALI`.

### 4. Cross-validation and choice of the epochs

```
python cv_regione_anno.py > log/cv_regione_anno.log
```

There are 9 folds (3 regions × 3 years). Piedmont is never used. The script writes:

- `modello/epoche_cv.json`, read by step 5;
- `risultati/cv_regione_anno_risultati.csv`.

### 5. Final model

```
python train_downscaling_head.py                    # all 4 variables (--parameter tx3 for one only)
python train_downscaling_head.py --valuta_test      # ONCE only, when the model is final
```

Trains on the whole training table with the epochs from step 4 and saves
`modello/checkpoints/<variable>_finale.pt`. Each file contains the ensemble of networks, the
features and their normalisation.

### 6. Maps and evaluation

```
python mappa_downscaling.py 2021-08-15 2021-10-04 2021-12-31   # ~1 km maps, needs the fields of those days
python mappa_rmse.py                                            # per-station error on the test (Piedmont 2020)
```

Maps go to `mappe/` and the comparisons with the stations to `risultati/`.

## Results

RMSE on the test set (Piedmont 2020), evaluated once:

| Variable | Aurora | Model | Reduction | Model in cross-validation |
|---|---|---|---|---|
| tx3 (°C) | 4.70 | 2.29 | −51% | 2.34 |
| tn3 (°C) | 3.77 | 1.94 | −49% | 2.26 |
| rr7 (mm) | 8.88 | 5.78 | −35% | 6.43 |
| qq2 (W/m²) | 62.67 | 39.68 | −37% | 34.10 |

## Experiments (`experiments/`)

Each experiment uses the same region × year cross-validation (`cv_comune.py`) and changes only one
thing with respect to the reference model. Results go to `risultati/`. Experiments are run as
modules from the main folder:

```
python -m experiments.esperimento_terreno > log/esperimento_terreno.log
```

| Script | Question |
|---|---|
| `esperimento_bilineare` | Aurora bilinearly interpolated instead of the nearest cell? |
| `esperimento_idw` | IDW for precipitation and radiation instead of bilinear? |
| `esperimento_terreno` | terrain descriptors in addition to elevation? (requires `python descrittori_terreno.py`) |
| `esperimento_pioggia` | target transformations for precipitation (square root, log)? |
| `esperimento_iperparametri` | network size, dropout, learning rate |
| `esperimento_ensemble` | average of several networks with different seeds |
| `esperimento_feature_ridotte` | can redundant features be removed? |
| `esperimento_importanza` | permutation importance of the final model |
| `esperimento_convoluzionale`, `_reg` | convolutional network on the Aurora window around the point |
| `esperimento_sarah_c` | radiation pre-training on SARAH-3 pixels |

`riferimenti_semplici.py` (in the main folder) compares the network with some simple corrections:
elevation correction, monthly mean, linear regression.

## What is not in the repository

Data, Aurora fields, tables and DEM are excluded by `.gitignore` because they are large or can be
recreated with steps 1–3. Code, notebooks, results (`.csv`), logs, `.png` maps and the networks of
the final model (`modello/checkpoints/*_finale.pt`) are included.

## Data licences

ECA&D data are for non-commercial research and education only
([data policy](https://www.ecad.eu/documents/ECAD_datapolicy.pdf)); the trained models depend on
them. Please cite ECA&D as required by its policy.
