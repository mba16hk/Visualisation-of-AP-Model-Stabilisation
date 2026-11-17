"""Cardiac AP model stabilisation browser.

    py -3 -m streamlit run app.py      (or double-click run_app.bat)
"""
from __future__ import annotations

import sys

if sys.version_info < (3, 12):
    sys.exit(f"The app needs Python 3.12 or newer (this is {sys.version.split()[0]}).")

from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import streamlit as st  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import config, io, plots, render  # noqa: E402
from core.labels import (BARS_MODELS, BIOMARKER_KEYS, BIOMARKER_TEXT,  # noqa: E402
                         CYCLE_LENGTHS, MODELS, TOL_SHORT, TOLERANCES, display_name,
                         model_variants)
from core.render import slug  # noqa: E402

st.set_page_config(page_title="Stabilisation Results", page_icon="🫀", layout="wide")

ALL = "All models"
CELL_OPTIONS = ["All", "ENDO", "EPI", "M"]
TOL_KEYS = list(TOLERANCES)                 # ["fine", "coarse"]
EPS_ORDER = ["5", "1", "0.5", "0.1"]        # Strategy B ε (percent), 1 % = default
PT_GLOBAL = "Global Convergence Analysis"
PT_STATE = "Time to Stabilisation per State"
PT_CLASS = "Functional-Class Stabilisation Heatmaps"


def model_text(m: str) -> str:
    return m if m == ALL else display_name(m)
VERSION = config.data_version() or "unknown"


# ------------------------------------------------------------ cached catalog
@st.cache_data(show_spinner=False)
def paced_catalog(version: str) -> dict:
    """{tol: {"models": [...], "cli_models": [...]}} for the built paced tables."""
    out = {}
    for tol in TOL_KEYS:
        df = io.paced_table(tol)
        if df is None:
            continue
        models = sorted(set(df["model"].astype(str)))
        cli = []
        if "Cli" in df.columns:
            cli = sorted(set(df.loc[df["Cli"].notna(), "model"].astype(str)))
        out[tol] = {"models": models, "cli_models": cli}
    return out


@st.cache_data(show_spinner=False)
def unpaced_catalog(version: str) -> dict:
    out = {}
    for tol in TOL_KEYS:
        st_ = io.unpaced_stats(tol)
        if st_ is not None:
            out[tol] = sorted(set(st_["model"].astype(str)))
    return out


@st.cache_data(show_spinner=False)
def model_states(version: str, model: str, variants: tuple[str, ...]) -> dict[str, str]:
    """{state: class} for the states Strategy A plots for this model (all tolerances)."""
    states: dict[str, str] = {}
    for v in variants:
        cmap = plots.state_class_map(model, v)
        for tol in TOL_KEYS:
            for s in plots.available_states(tol, model, v):
                states.setdefault(s, cmap.get(s, "Unclassified"))
    return states


def cells_for(choice: str) -> list[str]:
    return ["EPI", "ENDO", "M"] if choice == "All" else [choice]


def tolerance_control(single: bool, key: str) -> list[str]:
    """Fine/Coarse for all models; Fine/Coarse/Both for one model (single=True)."""
    if single:
        choice = st.radio("Tolerance", ["Fine", "Coarse", "Both"], horizontal=True,
                          key=f"{key}_one",
                          help="Fine = rtol 1E-6, atol 1E-8; Coarse = rtol 1E-4, atol 1E-6")
    else:
        choice = st.radio("Tolerance", ["Fine", "Coarse"], horizontal=True, key=f"{key}_all",
                          help="Fine = rtol 1E-6, atol 1E-8; Coarse = rtol 1E-4, atol 1E-6")
    return {"Fine": ["fine"], "Coarse": ["coarse"], "Both": ["fine", "coarse"]}[choice]


def bars_control(models: list[str], key: str) -> str:
    if any(m in BARS_MODELS for m in models):
        return st.radio(f"β-ARS variant (Doste, {display_name('Morotti')})", ["Both", "BARS", "noBARS"],
                        horizontal=True, key=key,
                        format_func=lambda o: {"Both": "Both", "BARS": "β-ARS",
                                               "noBARS": "no β-ARS"}[o])
    return "Both"


# ------------------------------------------------------------------- sidebar
render.export_settings()
with st.sidebar:
    with st.expander("Data source", expanded=False):
        st.caption(f"Data root: `{config.DATA_ROOT}`")
        st.caption("Data version:\n\n" + VERSION.replace("\n", "\n\n"))
        if st.button("Clear caches", help="Re-read the data after it has been rebuilt."):
            st.cache_data.clear()
            io.clear_caches()
            st.rerun()

st.title("Cardiac AP model stabilisation")
if not config.has_data():
    st.error(f"No data found under `{config.DATA_ROOT}`. Download "
             "`stabilisation-data.zip` from the repository's Releases page and "
             "unzip it into a `data` folder next to `app.py`, or run "
             "`build_data.py` (see README).")
    st.stop()

tab_paced, tab_unpaced = st.tabs(["Paced protocol", "Unpaced protocol"])

# ===================================================================== paced
with tab_paced:
    cat = paced_catalog(VERSION)
    c1, c2, c3, c4 = st.columns([1.2, 1, 1.6, 1.6])
    with c1:
        p_model = st.selectbox("Model", [ALL] + MODELS, key="p_model", format_func=model_text)
    single = p_model != ALL
    sel_models = [p_model] if single else MODELS
    with c2:
        p_cell = st.selectbox("Cell type", CELL_OPTIONS, key="p_cell")
    with c3:
        p_tols = tolerance_control(single, "p_tol")
    with c4:
        p_bars = bars_control(sel_models, "p_bars")

    c5, c6 = st.columns([1, 3])
    with c5:
        st.markdown("Pacing cycle length")
        cc = st.columns(3)
        p_cls = [cl for i, cl in enumerate(CYCLE_LENGTHS)
                 if cc[i].checkbox(f"{cl} ms", value=True, key=f"p_cl_{cl}")]
    with c6:
        cli_models = set().union(*(set(v["cli_models"]) for v in cat.values())) if cat else set()
        options = [b for b in BIOMARKER_KEYS
                   if b != "Cli" or any(m in cli_models for m in sel_models)]
        p_bio = st.multiselect("Biomarkers", options, default=["APD90"],
                               format_func=lambda k: BIOMARKER_TEXT[k], key="p_biomarkers")

    missing = [TOL_SHORT[t] for t in p_tols if t not in cat]
    if missing:
        st.warning(f"No paced data built for: {', '.join(missing)}.")
    if not p_cls:
        st.info("Tick at least one pacing cycle length.")
    elif not p_bio:
        st.info("Choose at least one biomarker.")
    elif all(t not in cat for t in p_tols):
        pass
    else:
        base = {"cells": cells_for(p_cell), "bars": p_bars, "cls": p_cls, "version": VERSION}
        if single and not any(p_model in cat[t]["models"] for t in p_tols if t in cat):
            st.info(f"No paced data for {display_name(p_model)} at the chosen tolerance.")
        elif single:
            spec = {**base, "model": p_model, "biomarkers": p_bio, "tols": p_tols}
            stem = slug(f"paced_{p_model}_{'_'.join(TOL_SHORT[t] for t in p_tols)}_{p_cell}")
            render.show_figures([(f"{display_name(p_model)} — paced", "paced_model", spec,
                                  stem)],
                                "paced_figures.zip")
        else:
            items = []
            for b in p_bio:
                spec = {**base, "models": MODELS, "biomarker": b, "tol": p_tols[0]}
                stem = slug(f"paced_all_{b}_{TOL_SHORT[p_tols[0]]}_{p_cell}")
                items.append((f"{BIOMARKER_TEXT[b]} — all models ({TOL_SHORT[p_tols[0]]})",
                              "paced_biomarker", spec, stem))
            render.show_figures(items, "paced_figures.zip")
        notes = []
        if "Morotti" in sel_models:
            notes.append(f"{display_name('Morotti')}: the first 5 beats are hidden for the "
                         "AP-shape biomarkers (initial transient), as in "
                         "paced_stabilisation_plots.py.")
        if any(b in ("Nai", "Ki", "Cli") for b in p_bio):
            notes.append("[Na⁺]ᵢ, [K⁺]ᵢ, [Cl⁻]ᵢ are end-of-beat values (one per beat).")
        if "Cai" in p_bio:
            notes.append("[Ca²⁺]ᵢ is drawn from the Cai trace as two lines per cell type: "
                         "the per-beat peak (systolic) and trough (diastolic).")
        if notes:
            st.caption(" ".join(notes))

# =================================================================== unpaced
with tab_unpaced:
    ucat = unpaced_catalog(VERSION)
    u_type = st.radio("Plot type", [PT_GLOBAL, PT_STATE, PT_CLASS], horizontal=True,
                      key="u_type")
    per_state = u_type == PT_GLOBAL
    c1, c2, c3, c4 = st.columns([1.2, 1, 1.6, 1.6])
    with c1:
        u_model = st.selectbox("Model", [ALL] + MODELS, key="u_model", format_func=model_text)
    u_single = u_model != ALL
    u_models = [u_model] if u_single else MODELS
    with c2:
        u_cell = st.selectbox("Cell type", CELL_OPTIONS, key="u_cell", disabled=not per_state,
                              help=None if per_state else
                              "Not applicable: these results are per model, averaged "
                              "over the cell types.")
    with c3:
        u_tols = tolerance_control(u_single or not per_state,
                                   "u_tol" if per_state else "u_tol_sum")
    with c4:
        u_bars = bars_control(u_models, "u_bars")

    if per_state:
        states = None
        if u_single:
            variants = tuple(model_variants(u_model, u_bars))
            smap = model_states(VERSION, u_model, variants)
            with st.expander("State filter (default: all states)", expanded=False):
                classes = sorted(set(smap.values()))
                f1, f2 = st.columns(2)
                with f1:
                    sel_cls = st.multiselect("State classes", classes,
                                             key=f"u_classes_{u_model}")
                pool = [s for s, c in smap.items() if not sel_cls or c in sel_cls]
                with f2:
                    sel_states = st.multiselect("States", pool, key=f"u_states_{u_model}",
                                                help="Leave empty to plot every state of "
                                                     "the chosen classes.")
                if sel_states:
                    states = sel_states
                elif sel_cls:
                    states = pool

        missing = [TOL_SHORT[t] for t in u_tols if t not in ucat]
        if missing:
            st.warning(f"No unpaced data built for: {', '.join(missing)}.")
        if all(t not in ucat for t in u_tols):
            pass
        elif u_single and not any(u_model in ucat[t] for t in u_tols if t in ucat):
            st.info(f"No unpaced data for {display_name(u_model)} at the chosen tolerance.")
        elif states is not None and not states:
            st.info("No states match the filter.")
        else:
            base = {"cells": cells_for(u_cell), "bars": u_bars, "version": VERSION}
            if u_single:
                spec = {**base, "model": u_model, "tols": u_tols, "states": states}
                stem = slug(f"unpaced_GlobalConvergence_{display_name(u_model)}_"
                            f"{'_'.join(TOL_SHORT[t] for t in u_tols)}_{u_cell}")
                title = (f"{display_name(u_model)} — {PT_GLOBAL}"
                         + (f", {len(states)} selected states" if states else ""))
                render.show_figures([(title, "unpaced_model", spec, stem)],
                                    "unpaced_figures.zip")
            else:
                spec = {**base, "models": MODELS, "tol": u_tols[0], "n_points": 2500}
                stem = slug(f"unpaced_GlobalConvergence_all_{TOL_SHORT[u_tols[0]]}_{u_cell}")
                render.show_figures([(f"All models — {PT_GLOBAL} ({TOL_SHORT[u_tols[0]]})",
                                      "unpaced_all", spec, stem)], "unpaced_figures.zip")
            st.caption("Normalised deviation = |x − x_ss| / max|x − x_ss|, x_ss = mean over "
                       "the last 2 s; both from the full-resolution simulation. Trajectories "
                       "are downsampled for display. Non-evolving states are excluded as in "
                       "unpaced_stabilisation_plots.py; the panel title gives the state "
                       "count.")
    else:
        kind = "times" if u_type == PT_STATE else "classes"
        avail = [set(io.available_eps(t, kind)) for t in u_tols]
        eps_opts = [e for e in EPS_ORDER if all(e in a for a in avail)]
        eps_opts += sorted((set.intersection(*avail) - set(eps_opts)) if avail else set(),
                           key=float, reverse=True)
        if not eps_opts:
            missing = [TOL_SHORT[t] for t, a in zip(u_tols, avail) if not a]
            st.warning("No stabilisation-summary data built for: "
                       f"{', '.join(missing) or 'the chosen tolerances'} "
                       "(run `build_data.py --only summary`).")
        else:
            e1, e2 = st.columns([1.6, 2.4])
            with e1:
                if u_type == PT_STATE or len(eps_opts) > 1:
                    eps = st.radio("Stabilisation threshold ε", eps_opts, horizontal=True,
                                   index=eps_opts.index("1") if "1" in eps_opts else 0,
                                   format_func=lambda e: f"{e}%", key=f"u_eps_{kind}",
                                   help="A state is stabilised once its normalised "
                                        "deviation stays below ε. Only the ε values "
                                        "built for the chosen tolerance(s) are offered.")
                else:
                    eps = eps_opts[0]
            if u_type == PT_CLASS:
                with e2:
                    metric = st.radio("Statistic", ["Mean", "Max"], horizontal=True,
                                      key="u_metric",
                                      help="Per-class mean or maximum stabilisation time.")
            tol_tag = "_".join(TOL_SHORT[t] for t in u_tols)
            mtag = "all" if not u_single else display_name(u_model)
            base = {"models": u_models, "bars": u_bars, "tols": u_tols, "eps": eps,
                    "version": VERSION}
            if u_type == PT_STATE:
                stem = slug(f"unpaced_TimeToStabilisation_{mtag}_{tol_tag}_eps{eps}")
                render.show_figures([(f"{model_text(u_model)} — {PT_STATE} (ε = {eps}%)",
                                      "strategy_b", base, stem)], "unpaced_figures.zip")
                st.caption("Time for each state's normalised deviation to stay below ε, "
                           "averaged over the cell types (one dot per state). Red: > 1850 s "
                           "(non-converging); green: 0 s (instantaneous); grey: eventual. "
                           "Bracketed numbers = states per model, after the same "
                           "non-evolving-state exclusions as the global analysis.")
            else:
                spec = {**base, "metric": metric}
                stem = slug(f"unpaced_ClassHeatmap_{metric}_{mtag}_{tol_tag}_eps{eps}")
                render.show_figures([(f"{model_text(u_model)} — {PT_CLASS} ({metric}, "
                                      f"ε = {eps}%)", "class_heatmap", spec, stem)],
                                    "unpaced_figures.zip")
                st.caption(f"{metric} stabilisation time per functional class of states "
                           "(colour, 0–2000 s in 8 bins); the number in each cell is the "
                           "state count, the bracketed number above each column the total. "
                           "Average TS = mean of the per-class values; Weighted Average TS "
                           "weights them by state count.")
