"""Reading the built data (see config.py for the layout) + shared helpers used by
build_data.py.

All loaders are plain functions keyed on (path, mtime) through lru_cache so they
work with and without Streamlit; app.py wraps the expensive ones in
st.cache_data as well.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .labels import CELL_FILE


# ---------------------------------------------------------------- helpers
def grid_indices(time: np.ndarray, n_target: int = 20_000, min_step_ms: float = 100.0,
                 n_log: int = 400, t_log_min_ms: float = 1e-3) -> np.ndarray:
    """Row indices that sample a (sorted) time vector on a uniform grid.

    Uniform part: step = max(min_step_ms, span / n_target) -> ~n_target points.
    A log-spaced set of n_log targets from t_log_min_ms after the start is added
    so the fast initial transients (sub-second) keep their shape. Each target
    picks the first actual sample at/after it (no interpolation), and the first
    and last rows are always kept.
    """
    time = np.asarray(time, dtype=float)
    n = len(time)
    if n <= 2:
        return np.arange(n)
    t0, t1 = time[0], time[-1]
    span = t1 - t0
    step = max(min_step_ms, span / n_target)
    targets = np.arange(t0, t1, step)
    if span > t_log_min_ms * 10:
        targets = np.union1d(targets, t0 + np.geomspace(t_log_min_ms, span, n_log))
    idx = np.searchsorted(time, targets, side="left")
    idx = np.clip(idx, 0, n - 1)
    return np.unique(np.concatenate([[0], idx, [n - 1]]))


def per_beat_end(time_ms: np.ndarray, values: np.ndarray, cl: float) -> np.ndarray:
    """One value per FULL beat: the last sample of beat n = [n*CL, (n+1)*CL)
    (end-diastolic). Mirrors determine_ionic_stabilisation_cycles._per_beat_trace
    with reduce='end' (beat = floor(time / CL), trailing partial beat dropped,
    NaN for an empty beat)."""
    time_ms = np.asarray(time_ms, dtype=float)
    values = np.asarray(values, dtype=float)
    n_beats = int(time_ms.max() // cl)
    out = np.full(max(n_beats, 0), np.nan)
    if n_beats <= 0:
        return out
    ends = np.searchsorted(time_ms, (np.arange(n_beats) + 1) * cl, side="left") - 1
    starts = np.arange(n_beats) * cl
    ok = (ends >= 0) & (time_ms[np.clip(ends, 0, None)] >= starts)
    out[ok] = values[ends[ok]]
    return out


def unpaced_file(tol: str, model: str, cell: str, variant: str) -> Path:
    suffix = f"_{variant}" if variant else ""
    return config.UNPACED_DIR / tol / model / f"{model}_{CELL_FILE[cell]}{suffix}.parquet"


# ---------------------------------------------------------------- loaders
def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.is_file() else -1.0


@lru_cache(maxsize=4)
def _paced(path: Path, mtime: float) -> pd.DataFrame | None:
    if mtime < 0:
        return None
    return pd.read_parquet(path)


def paced_table(tol: str) -> pd.DataFrame | None:
    """Per-beat table for one tolerance: model, variant, cell, cl, Cycle, biomarkers."""
    path = config.PACED_DIR / f"paced_{tol}.parquet"
    return _paced(path, _mtime(path))


@lru_cache(maxsize=4)
def _paced_groups(path: Path, mtime: float) -> dict:
    df = _paced(path, mtime)
    if df is None:
        return {}
    return {tuple(str(x) if not isinstance(x, (int, np.integer)) else int(x) for x in k): g
            for k, g in df.groupby(["model", "variant", "cell", "cl"], observed=True)}


def paced_group(tol: str, model: str, variant: str, cell: str, cl: int) -> pd.DataFrame | None:
    """Rows of one (model, variant, cell, cl) series, or None."""
    path = config.PACED_DIR / f"paced_{tol}.parquet"
    return _paced_groups(path, _mtime(path)).get((model, variant, cell, int(cl)))


@lru_cache(maxsize=4)
def _stats(path: Path, mtime: float) -> pd.DataFrame | None:
    if mtime < 0:
        return None
    return pd.read_parquet(path)


def unpaced_stats(tol: str) -> pd.DataFrame | None:
    """model, variant, cell, state, steady, max_dev (computed on the FULL data)."""
    path = config.UNPACED_DIR / tol / "stats.parquet"
    return _stats(path, _mtime(path))


@lru_cache(maxsize=48)
def _unpaced(path: Path, mtime: float) -> pd.DataFrame | None:
    if mtime < 0:
        return None
    return pd.read_parquet(path)


def unpaced_states(tol: str, model: str, cell: str, variant: str) -> pd.DataFrame | None:
    path = unpaced_file(tol, model, cell, variant)
    return _unpaced(path, _mtime(path))


@lru_cache(maxsize=2)
def _classes(path: Path, mtime: float) -> pd.DataFrame:
    if mtime < 0:
        return pd.DataFrame(columns=["model", "variant", "state", "class"])
    return pd.read_parquet(path)


def state_classes() -> pd.DataFrame:
    return _classes(config.CLASSES_FILE, _mtime(config.CLASSES_FILE))


# ------------------------------------------------- paced ion traces (Cai)
def trace_file(tol: str, model: str) -> Path:
    return config.TRACES_DIR / tol / f"{model}.parquet"


@lru_cache(maxsize=24)
def _traces(path: Path, mtime: float) -> dict:
    if mtime < 0:
        return {}
    df = pd.read_parquet(path)
    return {(str(v), str(c), int(cl)): g
            for (v, c, cl), g in df.groupby(["variant", "cell", "cl"], observed=True)}


def paced_trace(tol: str, model: str, variant: str, cell: str, cl: int) -> pd.DataFrame | None:
    """Thinned ion time series (time [ms], Cai, ...) of one paced run, or None."""
    path = trace_file(tol, model)
    return _traces(path, _mtime(path)).get((variant, cell, int(cl)))


# ------------------------------------- unpaced per-unit summaries (Strategy B / heatmaps)
def eps_tag(eps: str) -> str:
    """'0.5' -> file-name tag 'eps0.5'."""
    return f"eps{eps}"


def summary_file(tol: str, kind: str, eps: str) -> Path:
    """kind = 'times' (per-state times) or 'classes' (per-class summary)."""
    return config.UNPACED_DIR / tol / "summary" / f"{kind}_{eps_tag(eps)}.parquet"


def available_eps(tol: str, kind: str) -> list[str]:
    """ε values (percent, as strings, e.g. '1', '0.5') built for this tolerance."""
    folder = config.UNPACED_DIR / tol / "summary"
    if not folder.is_dir():
        return []
    out = [f.stem.split("_eps", 1)[1] for f in folder.glob(f"{kind}_eps*.parquet")]
    return sorted(out, key=float, reverse=True)


@lru_cache(maxsize=16)
def _plain(path: Path, mtime: float) -> pd.DataFrame | None:
    if mtime < 0:
        return None
    return pd.read_parquet(path)


def stab_times(tol: str, eps: str) -> pd.DataFrame | None:
    """unit, model, variant, state, time_ms, class (categorised_time_to_stabilisation)."""
    path = summary_file(tol, "times", eps)
    return _plain(path, _mtime(path))


def class_summary(tol: str, eps: str) -> pd.DataFrame | None:
    """class, order, unit, model, variant, mean_ms, max_ms, count
    (stabilisation_summary_by_class)."""
    path = summary_file(tol, "classes", eps)
    return _plain(path, _mtime(path))


def clear_caches() -> None:
    for f in (_paced, _paced_groups, _stats, _unpaced, _classes, _traces, _plain):
        f.cache_clear()
