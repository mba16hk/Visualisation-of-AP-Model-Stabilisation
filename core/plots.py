"""Figure builders. Every figure is built from a JSON-able spec dict so it can be
cached on its spec (see render.py). Return None when the selection has no data.

Paced figures mirror scripts/paced_stabilisation_plots.py (Set2 colours per cell
type, legend EPI/ENDO/M, panel titles "{CL} ms", x-label "Number of Cycles",
y-range shared across the CL columns of a row and across the β-ARS / no β-ARS
rows of the same model).

Unpaced figures reproduce "Strategy A" of scripts/unpaced_stabilisation_plots.py:
normalised deviation |x - x_ss| / max|x - x_ss| per state vs time, x_ss = mean
over the last 2000 ms, one colour per cell type, with the same per-model state
drop lists. x_ss and max|x - x_ss| come from the FULL simulation (precomputed by
build_data.py); only the plotted trajectories are downsampled.

Strategy B (time to stabilisation per state, scatter) and the functional-class
heatmaps are ported from StrategyB in scripts/unpaced_stabilisation_plots.py
and scripts/combined_tolerance_heatmaps.py; they read the converted workbooks
(io.stab_times / io.class_summary), one per tolerance and ε.

Reader-facing model names go through labels.display_name (Dutta -> CiPA,
Morotti -> hML); all logic stays keyed on the internal names.
"""
from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import matplotlib.transforms as mtransforms  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MultipleLocator, ScalarFormatter  # noqa: E402

from . import io  # noqa: E402
from .labels import (BIOMARKER_LABEL, CELL_TYPES, MOROTTI_SKIP_CYCLES,  # noqa: E402
                     MOROTTI_SKIP_FOR, TO_MICROMOLAR, TOL_LINESTYLE, TOL_SHORT,
                     TRACE_BIOMARKERS, UNIT_ORDER, cell_colors, display_name,
                     model_variants, unit_label)

GRID_KW = dict(linestyle="--", color="0.85", linewidth=0.8)


# ====================================================================== paced
def _paced_units(models: list[str], bars: str) -> list[tuple[str, str]]:
    return [(m, v) for m, v in UNIT_ORDER if m in models and v in model_variants(m, bars)]


def _paced_trace(tol: str, model: str, variant: str, cell: str, cl: int, metric: str):
    """Full (thinned) time series, x = time / CL so it shares the Cycle axis."""
    tr = io.paced_trace(tol, model, variant, cell, cl)
    if tr is None or metric not in tr.columns:
        return None
    tr = tr[["time", metric]].dropna()
    if tr.empty:
        return None
    y = tr[metric].to_numpy(dtype=float)
    if metric in TO_MICROMOLAR:
        y = y * 1000.0
    return tr["time"].to_numpy(dtype=float) / float(cl), y


def _paced_series(tol: str, model: str, variant: str, cell: str, cl: int, metric: str):
    if metric in TRACE_BIOMARKERS:
        return _paced_trace(tol, model, variant, cell, cl, metric)
    sub = io.paced_group(tol, model, variant, cell, cl)
    if sub is None or metric not in sub.columns:
        return None
    if sub.empty:
        return None
    sub = sub[["Cycle", metric]].dropna()
    if model == "Morotti" and metric in MOROTTI_SKIP_FOR:
        sub = sub[sub["Cycle"] >= MOROTTI_SKIP_CYCLES]
    if sub.empty:
        return None
    y = sub[metric].to_numpy(dtype=float)
    if metric in TO_MICROMOLAR:
        y = y * 1000.0
    return sub["Cycle"].to_numpy(dtype=float), y


def _paced_grid(rows: list[tuple[str, str, str, str]], cls: list[int], tols: list[str],
                cells: list[str], title: str | None = None):
    """rows: (row label, model, variant, metric). Columns = cls. Lines = cells
    (colour) x tols (line style)."""
    colors = cell_colors()
    # Gather data first; drop rows without any data.
    data, kept = {}, []
    for r, (_lab, model, variant, metric) in enumerate(rows):
        any_data = False
        for cl in cls:
            for tol in tols:
                for cell in cells:
                    s = _paced_series(tol, model, variant, cell, cl, metric)
                    if s is not None:
                        data[(r, cl, tol, cell)] = s
                        any_data = True
        if any_data:
            kept.append(r)
    if not kept:
        return None

    # y-range per (model, metric): shared across CL columns and BARS variants.
    yl = {}
    for (r, _cl, _tol, _cell), (_x, y) in data.items():
        key = (rows[r][1], rows[r][3])
        lo, hi = np.nanmin(y), np.nanmax(y)
        a, b = yl.get(key, (np.inf, -np.inf))
        yl[key] = (min(a, lo), max(b, hi))

    n_rows, n_cols = len(kept), len(cls)
    # Fixed-inch margins (constrained layout is slow for tall grids): title band
    # on top, x-label + legend band at the bottom so they never overlap.
    left_in, right_in, top_in, bot_in = 1.25, 0.25, (0.85 if title else 0.4), 1.05
    W = left_in + right_in + 3.3 * n_cols
    H = top_in + bot_in + 2.55 * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(W, H), sharex=True, squeeze=False)
    fig.subplots_adjust(left=left_in / W, right=1 - right_in / W, bottom=bot_in / H,
                        top=1 - top_in / H, hspace=0.32, wspace=0.08)
    for i, r in enumerate(kept):
        label, model, variant, metric = rows[r]
        lo, hi = yl[(model, metric)]
        pad = 0.04 * (hi - lo) if hi > lo else max(abs(hi) * 0.01, 1e-12)
        for j, cl in enumerate(cls):
            ax = axes[i][j]
            for tol in tols:
                for cell in cells:
                    s = data.get((r, cl, tol, cell))
                    if s is None:
                        continue
                    ls = TOL_LINESTYLE[tol] if len(tols) > 1 else "-"
                    if metric in TRACE_BIOMARKERS:
                        # Per-beat peak and trough lines: a filled envelope per cell
                        # type would hide the cell types drawn underneath it.
                        beat = np.floor(s[0]).astype(int)
                        env = pd.DataFrame({"b": beat, "y": s[1]}).groupby("b")["y"].agg(["min", "max"])
                        xb = env.index.to_numpy() + 0.5
                        ax.plot(xb, env["max"].to_numpy(), lw=1.0, color=colors[cell], linestyle=ls)
                        ax.plot(xb, env["min"].to_numpy(), lw=1.0, color=colors[cell], linestyle=ls)
                    else:
                        ax.plot(s[0], s[1], lw=1.3, color=colors[cell], linestyle=ls)
            ax.set_ylim(lo - pad, hi + pad)
            fmt = ScalarFormatter(useMathText=True, useOffset=False)
            fmt.set_scientific(False)
            ax.yaxis.set_major_formatter(fmt)
            if j != 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(label, fontsize=12, fontweight="bold", labelpad=6)
            ax.set_title(f"{cl} ms", fontsize=10, color="grey")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="x", labelbottom=(i == n_rows - 1), labelsize=9)
            ax.tick_params(axis="y", labelsize=9)
            ax.grid(True, alpha=0.2)

    fig.supxlabel("Number of Cycles", fontsize=15, fontweight="bold", y=0.62 / H, va="center")
    if title:
        fig.suptitle(title, fontsize=17, fontweight="bold", y=1 - 0.3 / H, va="center")
    present = [c for c in CELL_TYPES if any(k[3] == c for k in data)]
    handles = [Line2D([0], [0], color=colors[c], lw=2.5) for c in present]
    labels = list(present)
    if len(tols) > 1:
        handles += [Line2D([0], [0], color="0.3", lw=1.8, linestyle=TOL_LINESTYLE[t]) for t in tols]
        labels += [TOL_SHORT[t] for t in tols]
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.02 / H),
               ncol=len(labels), frameon=False, fontsize=13)
    return fig


def paced_biomarker(spec: dict):
    """All selected models for ONE biomarker: rows = model units, cols = CLs."""
    metric = spec["biomarker"]
    units = _paced_units(spec["models"], spec["bars"])
    rows = [(unit_label(m, v), m, v, metric) for m, v in units]
    return _paced_grid(rows, spec["cls"], [spec["tol"]], spec["cells"],
                       title=BIOMARKER_LABEL[metric])


def paced_model(spec: dict):
    """One model, several biomarkers: rows = biomarker (x β-ARS variant)."""
    model = spec["model"]
    variants = model_variants(model, spec["bars"])
    rows = []
    for metric in spec["biomarkers"]:
        for v in variants:
            lab = BIOMARKER_LABEL[metric]
            if v:
                lab = unit_label("", v).strip() + "\n" + lab
            rows.append((lab, model, v, metric))
    tol_txt = " vs ".join(TOL_SHORT[t] for t in spec["tols"])
    if len(spec["tols"]) > 1:
        tol_txt += " (solid = Fine, dashed = Coarse)"
    return _paced_grid(rows, spec["cls"], spec["tols"], spec["cells"],
                       title=f"{display_name(model)} — {tol_txt}")


# ==================================================================== unpaced
# Doste β-adrenergic / BARS-section states: constant when BARS is off, so they
# are dropped from the Doste WITHOUT-BARS plots (copied from the script).
_DOSTE_BARS_SECTION = [
    "ICaLp", "IKsp", "RyRp", "f_ikur", "f_inak", "f_tni", "ina_f_ina",
    "inhib1_p", "iup_f_plb",
    "cAMP_cav", "cAMP_cyt", "cAMP_eca",
    "PDE3_P_cav", "PDE3_P_cyt", "PDE4_P_cav", "PDE4_P_cyt", "PDE4_P_eca",
    "beta_cav_Gi_aGDP", "beta_cav_Gi_aGTP", "beta_cav_Gi_bg",
    "beta_cav_Gs_aGDP", "beta_cav_Gs_aGTP", "beta_cav_Gs_bg",
    "beta_cav_Rb1_grk_tot", "beta_cav_Rb1_pka_tot",
    "beta_cav_Rb2_grk_tot", "beta_cav_Rb2_pka_tot",
    "beta_cyt_Gs_aGDP", "beta_cyt_Gs_aGTP", "beta_cyt_Gs_bg",
    "beta_cyt_Rb1_grk_tot", "beta_cyt_Rb1_pka_tot",
    "beta_eca_Gi_aGDP", "beta_eca_Gi_aGTP", "beta_eca_Gi_bg",
    "beta_eca_Gs_aGDP", "beta_eca_Gs_aGTP", "beta_eca_Gs_bg",
    "beta_eca_Rb1_grk_tot", "beta_eca_Rb1_pka_tot",
    "beta_eca_Rb2_grk_tot", "beta_eca_Rb2_pka_tot",
    "pka_cav_A2R", "pka_cav_A2RC", "pka_cav_ARC", "pka_cav_C", "pka_cav_PKIC",
    "pka_cyt_A2R", "pka_cyt_A2RC", "pka_cyt_ARC", "pka_cyt_C", "pka_cyt_PKIC",
    "pka_eca_A2R", "pka_eca_A2RC", "pka_eca_ARC", "pka_eca_C", "pka_eca_PKIC",
]


def cols_to_drop_for(model: str, variant: str) -> set[str]:
    """State columns excluded from Strategy A (copied from
    unpaced_stabilisation_plots._cols_to_drop_for, keyed on model + variant)."""
    cols: set[str] = set()
    if model == "Morotti":
        cols.update([
            "INa_mk1", "INa_mk2", "INa_mk3", "INa_mk4", "INa_mk5", "INa_mk6",
            "INa_mk7", "INa_mk8", "INa_mk9", "INa_mk10", "INa_mk11", "INa_mk12",
            "rtos", "f", "fcaBj", "fcaBsl", "CaM_ecc", "LCC_PKAp", "RyR2809p",
            "Ca2CaMB_dyad", "Ca4CaMB_dyad", "CaMB_dyad", "Ki",
        ])
    if model == "Doste" and variant == "noBARS":
        cols.update(_DOSTE_BARS_SECTION)
    if model == "GB":
        cols.update(["Ki", "rtos", "ICajuncint",
                     "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8",
                     "C9", "C10", "C11", "C12", "C13", "C14", "C15", "O1"])
    if model == "Dutta":
        cols.update(["Cbound", "D", "IObound", "Obound"])
    if model == "BPS":
        cols.update(["Cb_dutta", "D_dutta", "IOb_dutta", "Ob_dutta", "EGTA", "EGTAi"])
    if model == "BPSLand":
        cols.update(["Cb_dutta", "D_dutta", "IOb_dutta", "Ob_dutta",
                     "EGTA", "EGTAi", "Jrelnp", "Jrelp", "xrf", "xrs",
                     "_unused_X8", "_unused_X25", "_unused_X28", "_unused_X29",
                     "_unused_X30", "_unused_X32", "_unused_X33"])
    return cols


def available_states(tol: str, model: str, variant: str) -> list[str]:
    """States plotted by Strategy A for this unit (after the drop list)."""
    st = io.unpaced_stats(tol)
    if st is None:
        return []
    sub = st[(st["model"] == model) & (st["variant"] == variant)]
    drop = cols_to_drop_for(model, variant)
    return list(dict.fromkeys(s for s in sub["state"] if s not in drop))


def state_class_map(model: str, variant: str) -> dict[str, str]:
    cl = io.state_classes()
    sub = cl[(cl["model"] == model) & (cl["variant"] == variant)]
    return dict(zip(sub["state"], sub["class"]))


def _norm_dev_panel(ax, tol: str, model: str, variant: str, cells: list[str],
                    states: list[str] | None, n_points: int | None) -> int | None:
    """Draw Strategy A for one unit on `ax`. Returns the number of states drawn,
    or None if there is no data."""
    stats = io.unpaced_stats(tol)
    if stats is None:
        return None
    colors = cell_colors()
    drop = cols_to_drop_for(model, variant)
    n_states, drawn = None, False
    for cell in cells:
        df = io.unpaced_states(tol, model, cell, variant)
        if df is None:
            continue
        st = stats[(stats["model"] == model) & (stats["variant"] == variant) &
                   (stats["cell"] == cell)].set_index("state")
        cols = [c for c in df.columns if c != "time" and c not in drop and c in st.index]
        if states is not None:
            cols = [c for c in cols if c in set(states)]
        if n_states is None:
            n_states = len(cols)
        if not cols:
            continue
        t = df["time"].to_numpy()
        idx = (io.grid_indices(t, n_target=n_points, min_step_ms=0.0, n_log=200)
               if n_points else np.arange(len(t)))
        ts = t[idx] / 1000.0                                   # ms -> s
        vals = df[cols].to_numpy()[idx]
        ss = st.loc[cols, "steady"].to_numpy()
        md = st.loc[cols, "max_dev"].to_numpy()
        with np.errstate(all="ignore"):
            delta = np.abs(vals - ss) / md
        delta[:, ~(md >= 1e-15)] = 0.0                          # flat states -> 0
        segs = [np.column_stack([ts, delta[:, k]]) for k in range(len(cols))]
        ax.add_collection(LineCollection(segs, colors=[colors[cell]], linewidths=0.8,
                                         alpha=0.5))
        drawn = True
    if not drawn:
        return None if n_states is None else 0
    ax.set_xlim(0, 2000)
    ax.set_ylim(-0.03, 1.03)
    ax.xaxis.set_major_locator(MultipleLocator(250 if ax.get_figure().get_figwidth() > 8 else 500))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(True, which="major", **GRID_KW)
    return n_states


def _unpaced_legend(fig, cells):
    colors = cell_colors()
    present = [c for c in CELL_TYPES if c in cells]
    handles = [Line2D([], [], color=colors[c], lw=4) for c in present]
    fig.legend(handles, present, loc="outside upper center", ncol=len(present),
               frameon=False, fontsize=16)


def unpaced_all(spec: dict):
    """Strategy A grid: one panel per model unit (4 columns)."""
    tol, cells = spec["tol"], spec["cells"]
    units = [(m, v) for m, v in UNIT_ORDER if m in spec["models"]
             and v in model_variants(m, spec["bars"])]
    if not units:
        return None
    n_cols = min(4, len(units))
    n_rows = math.ceil(len(units) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 3.9 * n_rows + 0.8),
                             sharex=True, sharey=True, squeeze=False, layout="constrained")
    flat = axes.flatten()
    any_data = False
    for ax, (m, v) in zip(flat, units):
        n = _norm_dev_panel(ax, tol, m, v, cells, None, spec.get("n_points", 2500))
        ax.set_title(f"{unit_label(m, v)}  ({n or 0})", fontsize=15, fontweight="bold",
                     loc="left")
        ax.tick_params(labelsize=11)
        if n is None:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center",
                    va="center", color="grey")
        else:
            any_data = True
    for ax in flat[len(units):]:
        ax.axis("off")
    if not any_data:
        plt.close(fig)
        return None
    fig.supxlabel("Time (s)", fontsize=18, fontweight="bold")
    fig.supylabel("Normalised Deviation (unitless)", fontsize=18, fontweight="bold")
    _unpaced_legend(fig, cells)
    return fig


def unpaced_model(spec: dict):
    """One model: rows = β-ARS variants, columns = tolerances."""
    model, cells, tols = spec["model"], spec["cells"], spec["tols"]
    variants = model_variants(model, spec["bars"])
    states = spec.get("states")             # None = all (after drop list)
    fig, axes = plt.subplots(len(variants), len(tols),
                             figsize=(7.5 * len(tols) + 0.6, 4.6 * len(variants) + 0.9),
                             sharex=True, sharey=True, squeeze=False, layout="constrained")
    any_data = False
    for i, v in enumerate(variants):
        for j, tol in enumerate(tols):
            ax = axes[i][j]
            n = _norm_dev_panel(ax, tol, model, v, cells, states, spec.get("n_points"))
            head = unit_label(model, v)
            if len(tols) > 1:
                head += f" — {TOL_SHORT[tol]}"
            ax.set_title(f"{head}  ({n or 0})", fontsize=14, fontweight="bold", loc="left")
            ax.tick_params(labelsize=11)
            if n is None:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center",
                        va="center", color="grey")
            elif n > 0:
                any_data = True
    if not any_data:
        plt.close(fig)
        return None
    fig.supxlabel("Time (s)", fontsize=15, fontweight="bold")
    fig.supylabel("Normalised Deviation (unitless)", fontsize=15, fontweight="bold")
    _unpaced_legend(fig, cells)
    return fig


# ======================================================= Strategy B / heatmaps
# Column order of the Strategy B scatter and the class heatmaps (HEATMAP_ORDER
# in unpaced_stabilisation_plots.py), as (model, variant).
SUMMARY_ORDER = [("Doste", "BARS"), ("Doste", "noBARS"), ("Morotti", "BARS"),
                 ("Morotti", "noBARS"), ("TOR", ""), ("TOR-DCl", ""), ("Dutta", ""),
                 ("ORd", ""), ("BPS", ""), ("BPSLand", ""), ("TNNP06", ""), ("GB", "")]
NONCONVERGING_S = 1850.0          # red in Strategy B: time > 1850 s
PANEL_TITLES = {"fine": "Fine Tolerance", "coarse": "Coarse Tolerance"}
# Class rows floated to the top of the heatmaps (combined_tolerance_heatmaps.py).
PRIORITY_ROWS = ["V", "Voltage", "Ca Dynamics", "Na Dynamics", "K Dynamics",
                 "Ca flux/RyR Dynamics", "CICR", "I_CaL Gates", "I_Kr Gates"]


def _summary_units(models: list[str], bars: str) -> list[tuple[str, str]]:
    return [(m, v) for m, v in SUMMARY_ORDER if m in models and v in model_variants(m, bars)]


def _tick_label(model: str, variant: str) -> str:
    """Two-line column label: display name over the β-ARS variant."""
    name = display_name(model)
    if variant == "BARS":
        return name + "\n" + r"($\beta$-ARS)"
    if variant == "noBARS":
        return name + "\n" + r"(no $\beta$-ARS)"
    return name


def _panel_title(tol: str, tols: list[str], extra: str = "") -> str:
    head = PANEL_TITLES[tol]
    if len(tols) > 1:
        head = f"({'AB'[tols.index(tol)]})  {head}"
    return head + extra


def _strategy_b_data(tol: str, eps: str, units):
    """{unit: time_s array} after the Strategy B drop list, or None."""
    df = io.stab_times(tol, eps)
    if df is None:
        return None
    out = {}
    for m, v in units:
        sub = df[(df["model"] == m) & (df["variant"] == v)]
        if sub.empty:
            continue
        sub = sub[~sub["state"].isin(cols_to_drop_for(m, v))]
        out[(m, v)] = sub["time_ms"].to_numpy(dtype=float) / 1000.0
    return out or None


def _strategy_b_panel(ax, data: dict, units: list, title: str, legend: bool) -> None:
    """Scatter branch of StrategyB: one jittered column per unit."""
    rng = np.random.default_rng(seed=42)
    jitter = 0.35
    for i, u in enumerate(units):
        t = data.get(u, np.array([]))
        x = np.full(len(t), i) + rng.uniform(-jitter, jitter, size=len(t))
        colors = np.where(t == 0, "green", np.where(t > NONCONVERGING_S, "red", "grey"))
        ax.scatter(x, t, c=colors, alpha=0.7, s=25, edgecolors="none")
    ax.set_xticks(range(len(units)))
    ax.set_xticklabels([_tick_label(m, v) for m, v in units], size=13, fontweight="bold")
    ax.set_xlim(-0.6, len(units) - 0.4)
    ax.set_ylabel("Time to stabilisation (s)", size=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=17, fontweight="bold", loc="left", pad=30)

    all_t = np.concatenate([data[u] for u in units if u in data])
    ymax = np.nanmax(all_t) if all_t.size else 1.0
    for i, u in enumerate(units):
        if u in data:
            ax.text(i, ymax * 1.05, f"({len(data[u])})", ha="center", va="bottom",
                    fontsize=11, fontweight="bold")
    if legend:
        handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=c,
                          markeredgecolor="none", markersize=9, alpha=0.85, label=lab)
                   for c, lab in (("red", "Non-converging"),
                                  ("green", "Instantaneous convergence"),
                                  ("grey", "Eventual convergence"))]
        leg = ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.13),
                        ncol=3, fontsize=12, frameon=False)
        for txt in leg.get_texts():
            txt.set_fontweight("bold")

    # Per-unit category % table below the tick labels (offsets in inches).
    h_in = ax.get_position().height * ax.get_figure().get_figheight()
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    cat_colors = {"Non-converging": "#d62728", "Eventual": "0.35", "Instantaneous": "green"}
    row_y = {lab: -(0.85 + 0.27 * k) / h_in
             for k, lab in enumerate(("Non-converging", "Eventual", "Instantaneous"))}
    for lab, y in row_y.items():
        ax.text(-0.012, y, lab, transform=ax.transAxes, ha="right", va="center",
                fontsize=10, fontweight="bold", color=cat_colors[lab], clip_on=False)
    for i, u in enumerate(units):
        t = data.get(u)
        if t is None or len(t) == 0:
            continue
        n = len(t)
        inst = int((t == 0).sum())
        nonc = int((t > NONCONVERGING_S).sum())
        pct = {"Non-converging": 100.0 * nonc / n, "Eventual": 100.0 * (n - inst - nonc) / n,
               "Instantaneous": 100.0 * inst / n}
        for lab, y in row_y.items():
            ax.text(i, y, f"{pct[lab]:.1f}%", transform=trans, ha="center", va="center",
                    fontsize=9, fontweight="bold", color=cat_colors[lab], clip_on=False)


def strategy_b(spec: dict):
    """Time to stabilisation per state (Strategy B scatter); one panel per tolerance."""
    tols, eps = spec["tols"], spec["eps"]
    units_all = _summary_units(spec["models"], spec["bars"])
    panels = []
    for tol in tols:
        data = _strategy_b_data(tol, eps, units_all)
        if data:
            panels.append((tol, data))
    if not panels:
        return None
    units = [u for u in units_all if any(u in d for _t, d in panels)]
    n = len(panels)
    # Fixed-inch layout: legend/title band, axes, tick-label + % table band.
    top_in, ax_in, bot_in, gap_in = 1.25, 5.6, 1.9, 1.0
    W = max(11.5, 2.4 + 1.1 * len(units))
    H = top_in + n * ax_in + (n - 1) * (bot_in + gap_in) + bot_in
    fig = plt.figure(figsize=(W, H))
    left, right = 1.9 / W, 1 - 0.3 / W
    for k, (tol, data) in enumerate(panels):
        y_top = top_in + k * (ax_in + bot_in + gap_in)
        ax = fig.add_axes([left, 1 - (y_top + ax_in) / H, right - left, ax_in / H])
        _strategy_b_panel(ax, data, units, _panel_title(tol, tols, f", ε = {eps}%"),
                          legend=(k == 0))
    return fig


def _summary_matrix(tol: str, eps: str, metric: str, units):
    """(values [class x unit] in s, counts [class x unit], class order) or None."""
    df = io.class_summary(tol, eps)
    if df is None:
        return None
    col = {"Mean": "mean_ms", "Max": "max_ms"}[metric]
    df = df.assign(key=df["model"] + "|" + df["variant"])
    df = df[df["key"].isin({_ukey(u) for u in units})]
    if df.empty:
        return None
    vals = df.pivot_table(index="class", columns="key", values=col, aggfunc="first",
                          dropna=False) / 1000.0
    cnts = df.pivot_table(index="class", columns="key", values="count", aggfunc="first",
                          dropna=False)
    order = df.groupby("class")["order"].min().sort_values().index.tolist()
    return vals, cnts, order


def _ukey(unit: tuple[str, str]) -> str:
    return f"{unit[0]}|{unit[1]}"


def _format_subscripts(label: str) -> str:
    import re
    return re.sub(r"(\S+)_([^\s]+)", lambda m: f"{m.group(1)}$_{{{m.group(2)}}}$", label)


def _heatmap_panel(ax, vals, counts, cmap, norm, units, rows, title) -> None:
    """combined_tolerance_heatmaps._draw_panel without seaborn (pcolormesh)."""
    keys = [_ukey(u) for u in units]
    v = vals.reindex(index=rows, columns=keys)
    c = counts.reindex(index=rows, columns=keys)
    arr = v.to_numpy(dtype=float)
    ax.pcolormesh(np.arange(len(units) + 1), np.arange(len(rows) + 1),
                  np.ma.masked_invalid(arr), cmap=cmap, norm=norm,
                  edgecolors="#808080", linewidth=1.2)
    ax.set_xlim(0, len(units))
    ax.set_ylim(len(rows), 0)
    for i in range(len(rows)):
        for j in range(len(units)):
            n = c.iat[i, j]
            if np.isfinite(arr[i, j]) and pd.notna(n):
                ax.text(j + 0.5, i + 0.5, f"{n:.0f}", ha="center", va="center",
                        fontsize=11, fontweight="bold", color="black")
    ax.set_xticks(np.arange(len(units)) + 0.5)
    ax.set_xticklabels([_tick_label(m, va) for m, va in units], rotation=0, ha="center",
                       fontweight="bold")
    ax.set_yticks(np.arange(len(rows)) + 0.5)
    ax.set_yticklabels([_format_subscripts(r) for r in rows], fontweight="bold")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, fontsize=19, fontweight="bold", loc="left", pad=40)
    totals = c.sum(axis=0)
    for j, tot in enumerate(totals):
        ax.text(j + 0.5, -0.4, f"({0 if pd.isna(tot) else int(tot)})", ha="center",
                va="bottom", fontsize=12, fontweight="bold")
    # Average / weighted-average TS of this panel's own per-class values.
    h_in = ax.get_position().height * ax.get_figure().get_figheight()
    trans = ax.get_xaxis_transform()
    y_avg, y_wavg = -0.68 / h_in, -0.92 / h_in
    for j, u in enumerate(keys):
        col, cnt = v[u], c[u].fillna(0.0)
        ok = col.notna()
        if not ok.any():
            continue
        a = col[ok].mean()
        den = cnt[ok].sum()
        w = (col[ok] * cnt[ok]).sum() / den if den else float("nan")
        ax.text(j + 0.5, y_avg, f"{a:.0f} s", ha="center", va="center", transform=trans,
                fontsize=9, clip_on=False)
        if np.isfinite(w):
            ax.text(j + 0.5, y_wavg, f"{w:.0f} s", ha="center", va="center",
                    transform=trans, fontsize=9, clip_on=False)
    for y, lab in ((y_avg, "Average TS:"), (y_wavg, "Weighted Average TS:")):
        ax.text(-0.15, y, lab, ha="right", va="center", transform=trans, fontsize=9,
                fontweight="bold", fontstyle="italic", clip_on=False)


def class_heatmap(spec: dict):
    """Functional-class stabilisation heatmap (Mean or Max); one panel per tolerance,
    shared discrete GreenGoldRed scale (8 bins, 0-2000 s) and colorbar."""
    tols, eps, metric = spec["tols"], spec["eps"], spec["metric"]
    units_all = _summary_units(spec["models"], spec["bars"])
    panels = []
    for tol in tols:
        r = _summary_matrix(tol, eps, metric, units_all)
        if r is not None:
            panels.append((tol, *r))
    if not panels:
        return None
    units = [u for u in units_all if any(_ukey(u) in p[1].columns for p in panels)]
    # Shared row order: priority classes first, then the summary order (panel A
    # first, classes only in later panels appended); drop classes empty everywhere.
    all_rows = []
    for _t, vals, _c, order in panels:
        all_rows += [r for r in order if r not in all_rows]
    all_rows = [r for r in all_rows
                if any(r in p[1].index
                       and p[1].loc[r].reindex([_ukey(u) for u in units]).notna().any()
                       for p in panels)]
    top = [r for r in PRIORITY_ROWS if r in all_rows]
    rows = top + [r for r in all_rows if r not in top]

    n_bins, lo, hi = 8, 0, 2000
    base = LinearSegmentedColormap.from_list("GreenGoldRed",
                                             ["#1a9850", "#f1c232", "#d73027"], N=256)
    cmap = LinearSegmentedColormap.from_list("discrete_GreenGoldRed",
                                             base(np.linspace(0, 1, n_bins)), N=n_bins)
    bounds = np.linspace(lo, hi, n_bins + 1)
    norm = BoundaryNorm(bounds, cmap.N)

    n = len(panels)
    top_in, ax_in, bot_in, cbar_in = 1.5, 0.55 * len(rows), 1.35, 1.2
    left_in, right_in = 2.9, 0.4
    W = max(9.0, left_in + right_in + 1.15 * len(units))
    H = top_in + n * ax_in + n * bot_in + (n - 1) * 0.6 + cbar_in
    fig = plt.figure(figsize=(W, H))
    left, right = left_in / W, 1 - right_in / W
    for k, (tol, vals, cnts, _o) in enumerate(panels):
        y_top = top_in + k * (ax_in + bot_in + 0.6)
        ax = fig.add_axes([left, 1 - (y_top + ax_in) / H, right - left, ax_in / H])
        _heatmap_panel(ax, vals, cnts, cmap, norm, units, rows, _panel_title(tol, tols))
    fig.suptitle(f"{metric} stabilisation time per functional class (ε = {eps}%)",
                 fontsize=20, fontweight="bold", y=1 - 0.35 / H, va="center")
    cw = min(0.75, 7.0 / W)
    cax = fig.add_axes([0.5 - cw / 2, 0.55 / H, cw, 0.22 / H])
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, boundaries=bounds, ticks=bounds, orientation="horizontal")
    cbar.ax.set_xticklabels([f"{b:.0f}" for b in bounds])
    cbar.ax.tick_params(labelsize=12, width=1.5)
    for lab in cbar.ax.get_xticklabels():
        lab.set_fontweight("bold")
    cbar.outline.set_edgecolor("black")
    cbar.outline.set_linewidth(1.8)
    cbar.set_label("Time (s)", fontsize=13, fontweight="bold", labelpad=6)
    return fig


BUILDERS = {
    "paced_biomarker": paced_biomarker,
    "paced_model": paced_model,
    "unpaced_all": unpaced_all,
    "unpaced_model": unpaced_model,
    "strategy_b": strategy_b,
    "class_heatmap": class_heatmap,
}


def build(kind: str, spec: dict):
    return BUILDERS[kind](spec)
