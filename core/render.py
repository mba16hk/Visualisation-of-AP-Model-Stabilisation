"""Streamlit side of figure rendering: cached PNG previews, TIFF/ZIP downloads."""
from __future__ import annotations

import hashlib
import io
import json
import re
import threading
import zipfile

import matplotlib.pyplot as plt
import streamlit as st

from . import export, plots

EXPORT_DPI = 300
# Previews are rasterised to about this many pixels across, within these dpi.
PREVIEW_WIDTH_PX = 2000
PREVIEW_DPI_MIN = 80
PREVIEW_DPI_MAX = 200
JOURNAL_WIDTH_MM = 178.0

# pyplot keeps global state and is not thread-safe; Streamlit runs each browser
# session in its own thread, so every build+save happens under this lock.
_LOCK = threading.Lock()


def spec_json(spec: dict) -> str:
    return json.dumps(spec, sort_keys=True, default=str)


def _key(kind: str, sj: str) -> str:
    return hashlib.sha1(f"{kind}|{sj}".encode()).hexdigest()[:16]


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9.+-]+", "_", text).strip("_")


@st.cache_data(max_entries=64, show_spinner=False)
def render_png(kind: str, sj: str):
    """(png bytes, figure size in inches) or (None, None) when there is no data."""
    with _LOCK:
        fig = plots.build(kind, json.loads(sj))
        if fig is None:
            return None, None
        try:
            width_in = fig.get_figwidth()
            dpi = min(PREVIEW_DPI_MAX, max(PREVIEW_DPI_MIN, PREVIEW_WIDTH_PX / width_in))
            return export.fig_to_png(fig, dpi), tuple(fig.get_size_inches())
        finally:
            plt.close(fig)


@st.cache_data(max_entries=8, show_spinner=False)
def render_tiff(kind: str, sj: str, dpi: float, width_mm: float | None) -> bytes | None:
    with _LOCK:
        fig = plots.build(kind, json.loads(sj))
        if fig is None:
            return None
        try:
            return export.fig_to_tiff(fig, dpi, width_mm)
        finally:
            plt.close(fig)


def export_settings() -> None:
    """Sidebar TIFF options (global, not tied to a tab)."""
    with st.sidebar.expander("TIFF export", expanded=False):
        st.selectbox("Resolution (dpi)", [EXPORT_DPI, 500, 600], index=0, key="export_dpi")
        st.checkbox(f"Scale to journal width ({JOURNAL_WIDTH_MM:g} mm)", key="export_journal",
                    help="Rescales the whole figure (fonts included) so it prints "
                         "exactly 178 mm wide at the chosen dpi.")


def export_settings_state() -> tuple[float, float | None]:
    dpi = float(st.session_state.get("export_dpi", EXPORT_DPI))
    width_mm = JOURNAL_WIDTH_MM if st.session_state.get("export_journal") else None
    return dpi, width_mm


def show_figure(title: str, kind: str, spec: dict, stem: str) -> tuple[str, str, str] | None:
    """Preview one figure with its TIFF download. Returns (kind, spec_json, stem)
    if it rendered (for the ZIP option)."""
    sj = spec_json(spec)
    key = _key(kind, sj)
    with st.spinner(f"Building {title}…"):
        png, size = render_png(kind, sj)
    if png is None:
        st.info(f"{title}: no data for this selection.")
        return None

    st.markdown(f"**{title}**")
    st.image(png, width="stretch")

    dpi, width_mm = export_settings_state()
    d = dpi if width_mm is None else dpi * (width_mm / 25.4) / size[0]
    w_px, h_px = int(size[0] * d), int(size[1] * d)
    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button("Prepare TIFF", key=f"prep_{key}",
                     help=f"{w_px} × {h_px} px, {dpi:g} dpi, LZW-compressed"):
            with st.spinner("Rendering TIFF…"):
                try:
                    st.session_state[f"tiff_{key}"] = (
                        render_tiff(kind, sj, dpi, width_mm), dpi, width_mm)
                except export.FigureTooLarge as exc:
                    st.error(str(exc))
    with c2:
        held = st.session_state.get(f"tiff_{key}")
        if held and held[0] and held[1:] == (dpi, width_mm):
            st.download_button(f"Download {stem}.tiff", data=held[0],
                               file_name=f"{stem}.tiff", mime="image/tiff", key=f"dl_{key}")
        else:
            st.caption(f"TIFF will be {w_px} × {h_px} px at {dpi:g} dpi.")
    st.divider()
    return kind, sj, stem


def show_figures(items: list[tuple[str, str, dict, str]], zip_name: str) -> None:
    """Several figures one after another, plus a ZIP of all their TIFFs.
    items: (title, kind, spec, stem) per figure."""
    if len(items) > 1:
        st.caption(f"{len(items)} figures for this selection.")
    rendered = [r for r in (show_figure(*it) for it in items) if r]
    if len(rendered) < 2:
        return
    dpi, width_mm = export_settings_state()
    zip_key = "zip_" + _key(f"zip|{dpi}|{width_mm}", "|".join(sj for _k, sj, _s in rendered))
    if st.button(f"Prepare ZIP of all {len(rendered)} TIFFs", key=f"prep_{zip_key}"):
        buf = io.BytesIO()
        bar = st.progress(0.0)
        try:
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
                for i, (kind, sj, stem) in enumerate(rendered):
                    data = render_tiff(kind, sj, dpi, width_mm)
                    if data:
                        zf.writestr(f"{stem}.tiff", data)
                    bar.progress((i + 1) / len(rendered))
            st.session_state[zip_key] = buf.getvalue()
        except export.FigureTooLarge as exc:
            st.error(str(exc))
    if zip_key in st.session_state:
        st.download_button("Download all TIFFs (.zip)", data=st.session_state[zip_key],
                           file_name=zip_name, mime="application/zip", key=f"dl_{zip_key}")
