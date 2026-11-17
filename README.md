# Visualisation of AP Model Stabilisation

An interactive [Streamlit](https://streamlit.io) app for browsing the stabilisation
behaviour of ten human ventricular action-potential (AP) models under a **paced** and
an **unpaced** protocol. Every combination of model, cell type, pacing rate, β-adrenergic
(β-ARS) state, solver tolerance and output metric can be plotted on demand and
downloaded as a publication-quality TIFF (one figure at a time or as a ZIP).

The app accompanies the manuscript on stabilisation of ventricular AP models. The
simulation outputs are too multidimensional to show in full in the paper, so this
app is the gallery of all generated plots.

## Models

| Name in the app | Model | Cell types | β-ARS variants |
|---|---|---|---|
| BPS | Bartolucci–Passini–Severi 2020 | ENDO, EPI, M | – |
| BPSLand | BPS 2020 (hybrid_07) with Land et al. 2017 contraction coupling | ENDO, EPI, M | – |
| CiPA | Dutta et al. 2017 (O'Hara–Rudy CiPA v1) | ENDO, EPI, M | – |
| Doste | Doste et al. ToR-ORd-BARS (ToR-ORd with β-adrenergic signalling) | ENDO, EPI | with (\*) / without (†) |
| GB | Grandi–Bers 2010 | ENDO, EPI | – |
| hML | Morotti et al. human ventricular myocyte model (with β-adrenergic and CaMKII signalling) | ENDO, EPI | with (\*) / without (†) |
| ORd | O'Hara–Rudy 2011 | ENDO, EPI, M | – |
| TNNP06 | ten Tusscher–Panfilov 2006 | ENDO, EPI, M | – |
| TOR | ToR-ORd (Tomek et al. 2019) | ENDO, EPI, M | – |
| TOR-DCl | ToR-ORd with dynamic chloride (Tomek et al. 2020) | ENDO, EPI, M | – |

All runs were integrated with CVODE at two tolerances:
**Fine** (rtol 1E-6, atol 1E-8) and **Coarse** (rtol 1E-4, atol 1E-6).

## What the app shows

### Paced protocol (2000 beats at cycle lengths of 500, 1000 and 2000 ms)

Per-beat biomarkers over the whole run: APD30/50/60/90, triangulation, V<sub>rest</sub>,
V<sub>peak</sub>, AP amplitude, dV/dt<sub>max</sub>, Ca<sup>2+</sup> transient metrics, and
the intracellular ions [Na⁺]ᵢ, [K⁺]ᵢ, [Cl⁻]ᵢ (end-of-beat values) and [Ca²⁺]ᵢ (per-beat
peak and trough).

- **All models**: one figure per biomarker; rows = models (Doste and hML have one row
  per β-ARS variant), columns = cycle lengths, colours = cell types.
- **One model**: rows = biomarkers. Choosing *Both* tolerances overlays Fine (solid)
  and Coarse (dashed).

### Unpaced protocol (2000 s without stimulus)

Three plot types:

- **Global Convergence Analysis**: the normalised deviation
  |x − x<sub>ss</sub>| / max|x − x<sub>ss</sub>| of every state variable over time
  (x<sub>ss</sub> = mean over the last 2 s), coloured by cell type. In single-model
  view the states can be filtered by functional class or individually.
- **Time to Stabilisation per State**: one point per state variable, showing the time
  after which its normalised deviation stays below ε. Red = non-converging (> 1850 s),
  green = instantaneous (0 s), grey = eventual. ε can be set to 5 %, 1 % (default),
  0.5 % or 0.1 %.
- **Functional-Class Stabilisation Heatmaps**: mean or maximum stabilisation time per
  functional class of state variables (e.g. Na dynamics, Ca buffers, I<sub>CaL</sub>
  gates), with average and weighted-average rows per model.

## Installation

Python 3.12 or newer.

```bash
git clone https://github.com/mba16hk/Visualisation-of-AP-Model-Stabilisation.git
cd Visualisation-of-AP-Model-Stabilisation
python -m pip install -r requirements.txt
```

### Data

The app reads a compact, pre-processed copy of the simulation outputs (~630 MB,
Parquet). Download **`stabilisation-data.zip`** from the
[Releases](../../releases) page and unzip it into a folder named `data` next to
`app.py`:

```
Visualisation-of-AP-Model-Stabilisation/
  app.py
  data/
    DATA_VERSION
    paced/
    unpaced/
    state_classes.parquet
```

To keep the data elsewhere, set the environment variable `STAB_APP_DATA_ROOT`, or
create `settings.local.toml` next to `app.py` containing
`data_root = "path/to/data"`.

## Running the app

```bash
python -m streamlit run app.py
```

On Windows you can also double-click `run_app.bat`. The app opens at
<http://localhost:8501>. The first figure of a new selection takes a few seconds
(all-models unpaced views ≈ 10–20 s) and is cached afterwards. TIFF resolution
(300 dpi by default) and a 178 mm journal-width option are in the sidebar.

## Rebuilding the data from raw simulation outputs

`build_data.py` converts the raw simulation folders into the data layout above. It is
only needed if you have re-run the simulations; most users should download the release.

| Part | Source | Built as |
|---|---|---|
| paced | per-beat `*_paced_APparams.csv` + `*_paced_ions.parquet` for each tolerance | `paced/paced_<fine\|coarse>.parquet` |
| Cai traces | Cai column of each `*_paced_ions.parquet`, min/max-thinned to ~8000 points per run | `paced/ions_traces/<fine\|coarse>/<Model>.parquet` |
| unpaced | each `*_unpaced_states_2000s.parquet`, resampled to a 0.1 s grid (+ log-spaced early points); steady state and maximum deviation computed from the full data | `unpaced/<fine\|coarse>/…` and `stats.parquet` |
| summary | `categorised_time_to_stabilisation*.xlsx` and `stabilisation_summary_by_class.xlsx` | `unpaced/<fine\|coarse>/summary/` |
| classes | `state_class_mapping.xlsx` | `state_classes.parquet` |

```bash
python build_data.py --paced-source <folder with "paced rtol …" folders> \
                     --unpaced-source <folder with "unpaced rtol …" folders> \
                     --summary-source <folder with "unpaced_plots rtol …" folders> \
                     --dest data
```

Use `--only paced|cai|unpaced|summary|classes` to rebuild one part and `--models` to
restrict the unpaced rebuild to some models. Afterwards press *Data source → Clear
caches* in the app (or restart it).

## Repository layout

```
app.py              Streamlit page (Paced / Unpaced tabs)
build_data.py       raw simulation outputs -> data folder
core/config.py      data folder lookup
core/io.py          Parquet readers (resampling, per-beat helpers)
core/labels.py      model names, units, biomarker labels
core/plots.py       figure builders (paced grids, convergence, time to stabilisation, heatmaps)
core/render.py      cached previews, TIFF/ZIP downloads
core/export.py      figure -> PNG/TIFF bytes
```

## Citation

If you use this app or its data, please cite the accompanying manuscript
(reference to be added on publication).
