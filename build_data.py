"""Build the app's data package from the raw stabilisation outputs.

    py -3 build_data.py --paced-source "<SO>\\Old Data" --unpaced-source "<SO>\\Old Data"
    py -3 build_data.py --paced-source "<SO>" --only paced        # after new paced runs
    py -3 build_data.py --paced-source "<SO>" --only cai          # Cai traces only
    py -3 build_data.py --only summary                            # Strategy B / heatmap tables

--paced-source   folder containing "paced rtol 1E-4 atol 1E-6" and "paced rtol 1E-6 atol 1E-8"
--unpaced-source folder containing "unpaced rtol 1E-4 atol 1E-6" and "unpaced rtol 1E-6 atol 1E-8"
--summary-source folder containing the two "unpaced_plots rtol ..." folders
                 (default: <SO>/New Data)
--dest           data root (default: the app's configured data root, see core/config.py)

Sources are only READ (one file at a time). Outputs overwrite earlier builds
(written to a temp file, then renamed):

paced/paced_<fine|coarse>.parquet
    One row per (model, variant, cell, cl, Cycle). Columns: the 13 APparams
    columns (from *_paced_APparams.csv) + Nai, Ki, Cai[, Cli] reduced to one value
    per beat (beat = floor(time / CL), value at the end of the beat - mirrors
    scripts/determine_ionic_stabilisation_cycles.py with reduce='end').
paced/ions_traces/<fine|coarse>/<Model>.parquet
    The ion time series (time, Cai) of every paced run, thinned to ~8000 rows per
    run by keeping every n-th row (as paced_stabilisation_plots._load_unit_ions).
    Columns: variant, cell, cl, time, Cai. Built with --only paced or --only cai.
unpaced/<fine|coarse>/<Model>/<Model>_<Cell>[_BARS|_noBARS].parquet
    The state trajectories resampled onto a uniform time grid (max(0.1 s,
    span/20000) step, plus ~400 log-spaced points over the initial transient),
    picking actual samples (no interpolation). Read in row batches.
unpaced/<fine|coarse>/stats.parquet
    Per (model, variant, cell, state): steady = mean over the last 2000 ms and
    max_dev = max |x - steady|, both from the FULL data (as Strategy A in
    scripts/unpaced_stabilisation_plots.py computes them).
unpaced/<fine|coarse>/summary/times_eps<e>.parquet
    categorised_time_to_stabilisation[ (eps e%)].xlsx (one sheet per unit) in long
    form: unit, model, variant, state, time_ms, class. The unsuffixed workbook is
    e = 1 %. Built with --only summary (or all).
unpaced/<fine|coarse>/summary/classes_eps<e>.parquet
    stabilisation_summary_by_class[ (eps e%)].xlsx ("Mean stabilisation time",
    "Max stabilisation time", "State counts"): class, order, unit, model, variant,
    mean_ms, max_ms, count.
state_classes.parquet
    state -> Class from state_class_mapping.xlsx (Doste/Morotti dagger sheets =
    noBARS).
DATA_VERSION, build_info.json
    Build date + sources of each part.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import config  # noqa: E402
from core.io import grid_indices, per_beat_end  # noqa: E402
from core.labels import APPARAM_COLUMNS, ION_COLUMNS, MODELS, TOLERANCES  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]          # the "C# Models" folder
DEFAULT_MAPPING = ROOT / "state_class_mapping.xlsx"
DEFAULT_SUMMARY_SOURCE = ROOT / "highthrouput_outputs" / "Stabilisation_outputs" / "New Data"
TRACE_POINTS = 8000              # paced_stabilisation_plots.MAX_PLOT_POINTS
TRACE_COLUMNS = ["Cai"]
STEADY_WINDOW_MS = 2000.0
BATCH_ROWS = 100_000

PACED_RE = re.compile(r"^(?P<model>.+?)_(?P<cell>ENDO|EPI|MID)_(?P<cl>\d+)ms"
                      r"(?:_(?P<bars>BARS|noBARS))?_paced_APparams\.csv$")
IONS_RE = re.compile(r"^(?P<model>.+?)_(?P<cell>ENDO|EPI|MID)_(?P<cl>\d+)ms"
                     r"(?:_(?P<bars>BARS|noBARS))?_paced_ions\.parquet$")
SUMMARY_RE = re.compile(r"^(?P<kind>categorised_time_to_stabilisation|"
                        r"stabilisation_summary_by_class)"
                        r"(?: \(eps (?P<eps>[\d.]+)%\))?\.xlsx$")
UNPACED_RE = re.compile(r"^(?P<model>.+?)_(?P<cell>ENDO|EPI|MID)"
                        r"(?:_(?P<bars>BARS|noBARS))?_unpaced_states_2000s\.parquet$")


def _cell(c: str) -> str:
    return "M" if c == "MID" else c


def _write_parquet(df: pd.DataFrame, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    df.to_parquet(tmp, compression="zstd", index=False)
    os.replace(tmp, dest)


# ------------------------------------------------------------------ paced
def build_paced(source: Path, dest: Path, models: list[str]) -> dict:
    info = {}
    for tol, (_label, suffix) in TOLERANCES.items():
        src = source / f"paced {suffix}"
        if not src.is_dir():
            print(f"[paced] {tol}: {src} not found - skipped")
            continue
        frames = []
        for model in models:
            mdir = src / model
            if not mdir.is_dir():
                continue
            for csv in sorted(mdir.glob("*_paced_APparams.csv")):
                m = PACED_RE.match(csv.name)
                if not m or m["model"] != model:
                    print(f"  [warn] unrecognised file name {csv.name}")
                    continue
                cl = int(m["cl"])
                ap = pd.read_csv(csv)
                keep = ["Cycle"] + [c for c in APPARAM_COLUMNS if c in ap.columns]
                ap = ap[keep].copy()
                ions_path = csv.with_name(csv.name.replace("_paced_APparams.csv",
                                                          "_paced_ions.parquet"))
                if ions_path.is_file():
                    names = pq.ParquetFile(ions_path).schema_arrow.names
                    cols = ["time"] + [c for c in ION_COLUMNS if c in names]
                    ions = pq.read_table(ions_path, columns=cols).to_pandas()
                    ions = ions.sort_values("time", kind="stable")
                    t = ions["time"].to_numpy()
                    beats = {c: per_beat_end(t, ions[c].to_numpy(), cl) for c in cols[1:]}
                    n = min(len(v) for v in beats.values())
                    per_beat = pd.DataFrame({"Cycle": np.arange(n)})
                    for c, v in beats.items():
                        per_beat[c] = v[:n]
                    ap = ap.merge(per_beat, on="Cycle", how="outer")
                else:
                    print(f"  [warn] no ions file for {csv.name}")
                ap.insert(0, "cl", cl)
                ap.insert(0, "cell", _cell(m["cell"]))
                ap.insert(0, "variant", m["bars"] or "")
                ap.insert(0, "model", model)
                frames.append(ap)
        if not frames:
            print(f"[paced] {tol}: no files")
            continue
        table = pd.concat(frames, ignore_index=True)
        for c in ("model", "variant", "cell"):
            table[c] = table[c].astype("category")
        out = dest / "paced" / f"paced_{tol}.parquet"
        _write_parquet(table, out)
        print(f"[paced] {tol}: {len(frames)} files -> {out} "
              f"({len(table):,} rows, {out.stat().st_size / 1e6:.1f} MB)")
        info[tol] = {"source": str(src), "files": len(frames)}
    return info


def _minmax_thin(df: pd.DataFrame, col: str, n_points: int) -> pd.DataFrame:
    """Keep the min and max row of `col` in each of n_points/2 equal-count bins.

    Plain every-n-th-row thinning aliases against the beat period (striped panels);
    keeping each bin's extremes preserves the transient peaks and troughs."""
    n_bins = max(n_points // 2, 1)
    v = df[col].to_numpy()
    bins = np.arange(len(v)) * n_bins // len(v)
    keep = np.zeros(len(v), dtype=bool)
    starts = np.flatnonzero(np.r_[True, bins[1:] != bins[:-1]])
    for s, e in zip(starts, np.r_[starts[1:], len(v)]):
        seg = v[s:e]
        if np.all(np.isnan(seg)):
            keep[s] = True
            continue
        keep[s + np.nanargmin(seg)] = True
        keep[s + np.nanargmax(seg)] = True
    return df[keep]


def build_traces(source: Path, dest: Path, models: list[str]) -> dict:
    """Thinned Cai time series of every paced run (every n-th row, ~TRACE_POINTS)."""
    info = {}
    for tol, (_label, suffix) in TOLERANCES.items():
        src = source / f"paced {suffix}"
        if not src.is_dir():
            print(f"[cai] {tol}: {src} not found - skipped")
            continue
        n_files = 0
        for model in models:
            mdir = src / model
            if not mdir.is_dir():
                continue
            frames = []
            for f in sorted(mdir.glob("*_paced_ions.parquet")):
                m = IONS_RE.match(f.name)
                if not m or m["model"] != model:
                    print(f"  [warn] unrecognised file name {f.name}")
                    continue
                names = pq.ParquetFile(f).schema_arrow.names
                cols = ["time"] + [c for c in TRACE_COLUMNS if c in names]
                df = pq.read_table(f, columns=cols).to_pandas()
                df = df.sort_values("time", kind="stable")
                if len(df) > TRACE_POINTS:
                    df = _minmax_thin(df, "Cai", TRACE_POINTS)
                df = df.reset_index(drop=True)
                df.insert(0, "cl", int(m["cl"]))
                df.insert(0, "cell", _cell(m["cell"]))
                df.insert(0, "variant", m["bars"] or "")
                frames.append(df)
            if not frames:
                continue
            table = pd.concat(frames, ignore_index=True)
            for c in ("variant", "cell"):
                table[c] = table[c].astype("category")
            _write_parquet(table, dest / "paced" / "ions_traces" / tol / f"{model}.parquet")
            n_files += len(frames)
        print(f"[cai] {tol}: {n_files} runs -> {dest / 'paced' / 'ions_traces' / tol}")
        if n_files:
            info[tol] = {"source": str(src), "files": n_files}
    return info


# ---------------------------------------------------------------- summary
def _unit(sheet: str) -> tuple[str, str]:
    for v in ("BARS", "noBARS"):
        if sheet.endswith("_" + v):
            return sheet[: -len(v) - 1], v
    return sheet, ""


def _times_long(path: Path) -> pd.DataFrame:
    rows = []
    for sheet, df in pd.read_excel(path, sheet_name=None).items():
        if not {"state", "time_to_stabilisation_ms"}.issubset(df.columns):
            print(f"  [warn] {path.name}:{sheet} lacks state/time columns - skipped")
            continue
        model, variant = _unit(sheet)
        cls = df["Class"] if "Class" in df.columns else pd.Series(pd.NA, index=df.index)
        rows.append(pd.DataFrame({
            "unit": sheet, "model": model, "variant": variant,
            "state": df["state"].astype(str).str.strip(),
            "time_ms": pd.to_numeric(df["time_to_stabilisation_ms"], errors="coerce"),
            "class": cls.astype("string").str.strip(),
        }))
    return pd.concat(rows, ignore_index=True)


def _classes_long(path: Path) -> pd.DataFrame:
    book = pd.read_excel(path, sheet_name=None)
    parts = {}
    for sheet, col in (("Mean stabilisation time", "mean_ms"),
                       ("Max stabilisation time", "max_ms"), ("State counts", "count")):
        df = book[sheet]
        df = df.set_index(df.columns[0])
        df.index = df.index.astype(str).str.strip()
        df.index.name = "class"
        df.columns.name = "unit"
        parts[col] = df.apply(pd.to_numeric, errors="coerce")
    order = {c: i for i, c in enumerate(parts["mean_ms"].index)}
    long = None
    for col, df in parts.items():
        s = df.stack(future_stack=True).rename(col).reset_index()
        long = s if long is None else long.merge(s, on=["class", "unit"], how="outer")
    long["order"] = long["class"].map(order).fillna(len(order)).astype(int)
    mv = long["unit"].map(_unit)
    long["model"] = [m for m, _v in mv]
    long["variant"] = [v for _m, v in mv]
    return long[["class", "order", "unit", "model", "variant", "mean_ms", "max_ms", "count"]]


def build_summary(source: Path, dest: Path) -> dict:
    """Convert the per-tolerance Strategy B / class-summary workbooks (every eps
    variant present) to parquet."""
    info = {}
    for tol, (_label, suffix) in TOLERANCES.items():
        src = source / f"unpaced_plots {suffix}"
        if not src.is_dir():
            print(f"[summary] {tol}: {src} not found - skipped")
            continue
        out_dir = dest / "unpaced" / tol / "summary"
        built = []
        for f in sorted(src.glob("*.xlsx")):
            m = SUMMARY_RE.match(f.name)
            if not m:
                continue
            eps = f"{float(m['eps'] or '1'):g}"
            if m["kind"] == "categorised_time_to_stabilisation":
                kind, df = "times", _times_long(f)
            else:
                kind, df = "classes", _classes_long(f)
            _write_parquet(df, out_dir / f"{kind}_eps{eps}.parquet")
            built.append(f"{kind} eps={eps}%")
        print(f"[summary] {tol}: {', '.join(built) or 'nothing'}")
        if built:
            info[tol] = {"source": str(src), "files": len(built)}
    return info


# ---------------------------------------------------------------- unpaced
def _reduce_unpaced(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """(downsampled df, stats df [state, steady, max_dev], n_rows_full), reading
    the file in row batches so memory stays bounded for the 1.7 GB Morotti files."""
    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    states = [c for c in names if c != "time"]
    t = pf.read(columns=["time"]).column(0).to_numpy()
    if np.any(np.diff(t) < 0):
        raise ValueError(f"{path.name}: time is not sorted")
    n = len(t)
    keep_idx = grid_indices(t)
    tail_from = t.max() - STEADY_WINDOW_MS

    k = len(states)
    cmin = np.full(k, np.inf)
    cmax = np.full(k, -np.inf)
    tail_sum = np.zeros(k)
    tail_cnt = np.zeros(k)
    kept_parts = []
    offset = 0
    for batch in pf.iter_batches(batch_size=BATCH_ROWS, columns=names):
        arr = np.column_stack([batch.column(c).to_numpy(zero_copy_only=False).astype(float)
                               for c in states])
        bt = batch.column("time").to_numpy(zero_copy_only=False).astype(float)
        nb = len(bt)
        with np.errstate(all="ignore"):
            cmin = np.fmin(cmin, np.nanmin(arr, axis=0, initial=np.inf))
            cmax = np.fmax(cmax, np.nanmax(arr, axis=0, initial=-np.inf))
        tail = bt >= tail_from
        if tail.any():
            sub = arr[tail]
            tail_sum += np.nansum(sub, axis=0)
            tail_cnt += np.sum(~np.isnan(sub), axis=0)
        lo, hi = np.searchsorted(keep_idx, [offset, offset + nb])
        local = keep_idx[lo:hi] - offset
        if len(local):
            part = pd.DataFrame(arr[local], columns=states)
            part.insert(0, "time", bt[local])
            kept_parts.append(part)
        offset += nb
        del arr, batch
    steady = np.where(tail_cnt > 0, tail_sum / np.maximum(tail_cnt, 1), np.nan)
    max_dev = np.fmax(cmax - steady, steady - cmin)
    stats = pd.DataFrame({"state": states, "steady": steady, "max_dev": max_dev})
    return pd.concat(kept_parts, ignore_index=True), stats, n


def build_unpaced(source: Path, dest: Path, models: list[str]) -> dict:
    info = {}
    for tol, (_label, suffix) in TOLERANCES.items():
        src = source / f"unpaced {suffix}"
        if not src.is_dir():
            print(f"[unpaced] {tol}: {src} not found - skipped")
            continue
        stats_path = dest / "unpaced" / tol / "stats.parquet"
        old = pd.read_parquet(stats_path) if stats_path.is_file() else None
        all_stats, n_files = [], 0
        for model in models:
            mdir = src / model
            if not mdir.is_dir():
                continue
            for f in sorted(mdir.glob("*_unpaced_states_2000s.parquet")):
                m = UNPACED_RE.match(f.name)
                if not m or m["model"] != model:
                    print(f"  [warn] unrecognised file name {f.name}")
                    continue
                t0 = time.time()
                df, st, n_full = _reduce_unpaced(f)
                variant = m["bars"] or ""
                out = (dest / "unpaced" / tol / model /
                       f"{model}_{m['cell']}{'_' + variant if variant else ''}.parquet")
                _write_parquet(df, out)
                st.insert(0, "cell", _cell(m["cell"]))
                st.insert(0, "variant", variant)
                st.insert(0, "model", model)
                st["n_rows_full"] = n_full
                st["n_rows_kept"] = len(df)
                all_stats.append(st)
                n_files += 1
                print(f"[unpaced] {tol} {f.name}: {n_full:,} -> {len(df):,} rows, "
                      f"{out.stat().st_size / 1e6:.1f} MB ({time.time() - t0:.0f} s)")
        if not all_stats:
            continue
        stats = pd.concat(all_stats, ignore_index=True)
        if old is not None:   # keep stats of models not rebuilt this time
            old = old[~old["model"].isin(models)]
            stats = pd.concat([old, stats], ignore_index=True)
        _write_parquet(stats, stats_path)
        info[tol] = {"source": str(src), "files": n_files}
    return info


# ---------------------------------------------------------------- mapping
def build_classes(mapping: Path, dest: Path) -> dict:
    if not mapping.is_file():
        print(f"[classes] {mapping} not found - skipped")
        return {}
    book = pd.read_excel(mapping, sheet_name=None)
    rows = []
    for sheet, df in book.items():
        if not {"state", "Class"}.issubset(df.columns):
            continue
        name = sheet.strip()
        model, variant = name, ""
        for bm in ("Doste", "Morotti"):
            if name == bm:
                model, variant = bm, "BARS"
            elif name.startswith(bm):          # dagger sheet = no β-ARS
                model, variant = bm, "noBARS"
        sub = df[["state", "Class"]].dropna(subset=["state"]).copy()
        sub["state"] = sub["state"].astype(str).str.strip()
        sub["Class"] = sub["Class"].astype(str).str.strip()
        sub = sub.drop_duplicates(subset="state")
        sub.insert(0, "variant", variant)
        sub.insert(0, "model", model)
        rows.append(sub.rename(columns={"Class": "class"}))
    table = pd.concat(rows, ignore_index=True)
    _write_parquet(table, dest / "state_classes.parquet")
    print(f"[classes] {len(table)} state->class rows from {mapping.name}")
    return {"source": str(mapping)}


# ------------------------------------------------------------------- main
def write_version(dest: Path, new_info: dict) -> None:
    info_path = dest / "build_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.is_file() else {}
    info.update(new_info)
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    lines = []
    for part in ("paced", "cai", "unpaced", "summary", "classes"):
        p = info.get(part)
        if not p:
            continue
        built = p.get("built", "?")
        if part == "classes":
            lines.append(f"classes: built {built} from {p.get('source')}")
            continue
        srcs = "; ".join(f"{t}={v['source']} ({v['files']} files)"
                         for t, v in p.get("tolerances", {}).items())
        lines.append(f"{part}: built {built} from {srcs}")
    (dest / "DATA_VERSION").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--paced-source", type=Path,
                    help='folder containing the two "paced rtol ..." folders')
    ap.add_argument("--unpaced-source", type=Path,
                    help='folder containing the two "unpaced rtol ..." folders')
    ap.add_argument("--dest", type=Path, default=config.DATA_ROOT)
    ap.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING,
                    help="state_class_mapping.xlsx")
    ap.add_argument("--summary-source", type=Path, default=DEFAULT_SUMMARY_SOURCE,
                    help='folder containing the two "unpaced_plots rtol ..." folders')
    ap.add_argument("--only", choices=["all", "paced", "cai", "unpaced", "summary", "classes"],
                    default="all",
                    help="paced = per-beat table + Cai traces; cai = Cai traces only; "
                         "summary = Strategy B / heatmap workbooks")
    ap.add_argument("--models", nargs="*", default=MODELS,
                    help="restrict to these models (default: all 10)")
    args = ap.parse_args()

    dest = args.dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Destination: {dest}")
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    new_info = {}
    partial = sorted(args.models) != sorted(MODELS)
    if args.only in ("all", "paced"):
        if not args.paced_source:
            ap.error("--paced-source is required to build the paced data")
        if partial:
            print("[paced] NOTE: --models given; the paced tables will contain only those models")
        tinfo = build_paced(args.paced_source.resolve(), dest, args.models)
        if tinfo:
            new_info["paced"] = {"built": stamp, "tolerances": tinfo}
    if args.only in ("all", "paced", "cai"):
        if not args.paced_source:
            ap.error("--paced-source is required to build the Cai traces")
        tinfo = build_traces(args.paced_source.resolve(), dest, args.models)
        if tinfo:
            new_info["cai"] = {"built": stamp, "tolerances": tinfo}
    if args.only in ("all", "unpaced"):
        if not args.unpaced_source:
            ap.error("--unpaced-source is required to build the unpaced data")
        tinfo = build_unpaced(args.unpaced_source.resolve(), dest, args.models)
        if tinfo:
            new_info["unpaced"] = {"built": stamp, "tolerances": tinfo}
    if args.only in ("all", "summary"):
        tinfo = build_summary(args.summary_source.resolve(), dest)
        if tinfo:
            new_info["summary"] = {"built": stamp, "tolerances": tinfo}
    if args.only in ("all", "classes", "unpaced"):
        c = build_classes(args.mapping, dest)
        if c:
            new_info["classes"] = {"built": stamp, **c}
    write_version(dest, new_info)
    print((dest / "DATA_VERSION").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
