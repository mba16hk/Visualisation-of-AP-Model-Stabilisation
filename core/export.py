"""Figure -> bytes (PNG preview, TIFF download)."""
from __future__ import annotations

import io

# Matplotlib cannot rasterise beyond 2**16 px per side; keep well clear of the
# memory a huge RGBA buffer needs (4 bytes per pixel).
MAX_SIDE_PX = 60_000
MAX_PIXELS = 250_000_000


class FigureTooLarge(Exception):
    pass


def effective_dpi(fig, dpi: float, width_mm: float | None) -> float:
    """Raster dpi. With width_mm the image is scaled so it is exactly that wide
    when printed at `dpi`."""
    if width_mm is None:
        return dpi
    return dpi * (width_mm / 25.4) / fig.get_figwidth()


def pixel_size(fig, dpi: float, width_mm: float | None = None) -> tuple[int, int]:
    d = effective_dpi(fig, dpi, width_mm)
    w, h = fig.get_size_inches()
    return int(round(w * d)), int(round(h * d))


def fig_to_png(fig, dpi: float) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
    return buf.getvalue()


def fig_to_tiff(fig, dpi: float = 300, width_mm: float | None = None) -> bytes:
    """LZW-compressed TIFF tagged `dpi` dots per inch."""
    w, h = pixel_size(fig, dpi, width_mm)
    if max(w, h) > MAX_SIDE_PX or w * h > MAX_PIXELS:
        raise FigureTooLarge(
            f"{w} x {h} px ({w * h / 1e6:.0f} Mpx) is too large to render at "
            f"{dpi:g} dpi. Use the journal-width option or a lower dpi.")
    buf = io.BytesIO()
    fig.savefig(buf, format="tiff", dpi=effective_dpi(fig, dpi, width_mm),
                facecolor="white",
                pil_kwargs={"compression": "tiff_lzw", "dpi": (dpi, dpi)})
    return buf.getvalue()
