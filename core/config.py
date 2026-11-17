"""Where the app finds its (built) data.

Data root lookup order (first hit wins):
    1. environment variable STAB_APP_DATA_ROOT
    2. `data_root = "..."` in settings.local.toml (next to app.py)
    3. <app>/data

The data root is what build_data.py writes (--dest):
    DATA_VERSION
    paced/paced_<fine|coarse>.parquet          per-beat biomarkers + ions
    paced/ions_traces/<fine|coarse>/<Model>.parquet   thinned ion time series (Cai trace)
    unpaced/<fine|coarse>/<Model>/*.parquet    downsampled state trajectories
    unpaced/<fine|coarse>/stats.parquet        steady state + max deviation (full data)
    unpaced/<fine|coarse>/summary/times_eps<ε>.parquet    per-state stabilisation times
    unpaced/<fine|coarse>/summary/classes_eps<ε>.parquet  per-class mean/max/count
    state_classes.parquet                      state -> class (state_class_mapping.xlsx)
"""
from __future__ import annotations

import os
import tomllib
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent


def _settings() -> dict:
    path = APP_ROOT / "settings.local.toml"
    if path.is_file():
        with path.open("rb") as fh:
            return tomllib.load(fh)
    return {}


def _data_root() -> Path:
    env = os.environ.get("STAB_APP_DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    configured = _settings().get("data_root")
    if configured:
        return Path(configured).expanduser().resolve()
    return APP_ROOT / "data"


DATA_ROOT = _data_root()
PACED_DIR = DATA_ROOT / "paced"
UNPACED_DIR = DATA_ROOT / "unpaced"
TRACES_DIR = PACED_DIR / "ions_traces"
CLASSES_FILE = DATA_ROOT / "state_classes.parquet"


def data_version() -> str | None:
    stamp = DATA_ROOT / "DATA_VERSION"
    return stamp.read_text(encoding="utf-8").strip() if stamp.is_file() else None


def has_data() -> bool:
    return PACED_DIR.is_dir() or UNPACED_DIR.is_dir()
