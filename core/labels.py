"""App vocabulary: models, units, tolerances, biomarkers and their plot labels.

Plot labels are matplotlib mathtext. The resting potential is always V_rest and
the AP peak V_peak (never "RMP" / "AP_max").
"""
from __future__ import annotations

import matplotlib

MODELS = ["BPS", "BPSLand", "Doste", "Dutta", "GB", "Morotti", "ORd", "TNNP06",
          "TOR", "TOR-DCl"]
BARS_MODELS = {"Doste", "Morotti"}
# Names shown to the reader. Internal keys (folder / file / sheet names) keep
# the original model names; only what is drawn or displayed is renamed.
MODEL_DISPLAY = {"Dutta": "CiPA", "Morotti": "hML"}


def display_name(model: str) -> str:
    """Reader-facing model name (Dutta -> CiPA, Morotti -> hML)."""
    return MODEL_DISPLAY.get(model, model)


# Cell types as written in file names -> as shown in plots.
CELL_FILE = {"ENDO": "ENDO", "EPI": "EPI", "M": "MID"}
CELL_TYPES = ["EPI", "ENDO", "M"]          # legend order used by the scripts
CYCLE_LENGTHS = [500, 1000, 2000]

# Tolerance key -> (label, source folder suffix)
TOLERANCES = {
    "fine": ("Fine (rtol 1E-6, atol 1E-8)", "rtol 1E-6 atol 1E-8"),
    "coarse": ("Coarse (rtol 1E-4, atol 1E-6)", "rtol 1E-4 atol 1E-6"),
}
TOL_SHORT = {"fine": "Fine", "coarse": "Coarse"}
TOL_LINESTYLE = {"fine": "-", "coarse": "--"}

# A "unit" = one plotted row: (model, variant). variant is "" for models without
# a β-ARS switch, else "BARS" / "noBARS".
UNIT_ORDER = [("Doste", "BARS"), ("Doste", "noBARS"), ("Morotti", "BARS"),
              ("Morotti", "noBARS"), ("BPS", ""), ("BPSLand", ""), ("TOR", ""),
              ("TOR-DCl", ""), ("ORd", ""), ("Dutta", ""), ("TNNP06", ""), ("GB", "")]


def model_variants(model: str, bars_choice: str = "Both") -> list[str]:
    if model not in BARS_MODELS:
        return [""]
    return {"Both": ["BARS", "noBARS"], "BARS": ["BARS"], "noBARS": ["noBARS"]}[bars_choice]


def unit_label(model: str, variant: str) -> str:
    """Mathtext row label, e.g. 'Doste (β-ARS)' / 'Doste (no β-ARS)'."""
    model = display_name(model)
    if variant == "BARS":
        return rf"{model} ($\beta$-ARS)"
    if variant == "noBARS":
        return rf"{model} (no $\beta$-ARS)"
    return model


def unit_text(model: str, variant: str) -> str:
    """Plain-text version of unit_label (for captions / file names)."""
    model = display_name(model)
    return {"BARS": f"{model} β-ARS", "noBARS": f"{model} no β-ARS"}.get(variant, model)


def cell_colors() -> dict[str, tuple]:
    cmap = matplotlib.colormaps["Set2"]
    return {"EPI": cmap(0 / 3), "ENDO": cmap(1 / 3), "M": cmap(2 / 3)}


# Paced biomarkers: key = column in the built paced table.
# (key, selector text, plot label)
BIOMARKERS = [
    ("APD30", "APD30", r"APD$_{30}$ (ms)"),
    ("APD50", "APD50", r"APD$_{50}$ (ms)"),
    ("APD60", "APD60", r"APD$_{60}$ (ms)"),
    ("APD90", "APD90", r"APD$_{90}$ (ms)"),
    ("Triangulation 30", "Triangulation 30 (APD90 − APD30)", r"Triangulation$_{30}$ (ms)"),
    ("Triangulation 60", "Triangulation 60 (APD90 − APD60)", r"Triangulation$_{60}$ (ms)"),
    ("dV/dtmax steepest", "dV/dtmax", r"dV/dt$_{\mathrm{max}}$ (mV/ms)"),
    ("Resting Membrane Potential", "V_rest", r"V$_{\mathrm{rest}}$ (mV)"),
    ("AP_max", "V_peak", r"V$_{\mathrm{peak}}$ (mV)"),
    ("AP_amp", "AP amplitude", r"AP$_{\mathrm{amp}}$ (mV)"),
    ("Catmax", "Ca transient max", r"CaT$_{\mathrm{max}}$ ($\mu$M)"),
    ("Catmin", "Ca transient min", r"CaT$_{\mathrm{min}}$ ($\mu$M)"),
    ("Catamp", "Ca transient amplitude", r"CaT$_{\mathrm{amp}}$ ($\mu$M)"),
    ("Nai", "[Na+]i", r"[Na$^{+}$]$_{i}$ (mM)"),
    ("Ki", "[K+]i", r"[K$^{+}$]$_{i}$ (mM)"),
    ("Cai", "[Ca2+]i", r"[Ca$^{2+}$]$_{i}$ ($\mu$M)"),
    ("Cli", "[Cl−]i", r"[Cl$^{-}$]$_{i}$ (mM)"),
]
BIOMARKER_KEYS = [b[0] for b in BIOMARKERS]
BIOMARKER_TEXT = {b[0]: b[1] for b in BIOMARKERS}
BIOMARKER_LABEL = {b[0]: b[2] for b in BIOMARKERS}
APPARAM_COLUMNS = BIOMARKER_KEYS[:13]
ION_COLUMNS = ["Nai", "Ki", "Cai", "Cli"]
# Biomarkers plotted as the (thinned) full time series instead of one value per
# beat; x = time / CL so they share the Cycle axis.
TRACE_BIOMARKERS = {"Cai"}
# Displayed in µM (stored in mM).
TO_MICROMOLAR = {"Catmax", "Catmin", "Catamp", "Cai"}
# The paced script drops Morotti's first 5 beats for the AP-shape biomarkers
# (initial transient); mirrored here.
MOROTTI_SKIP_CYCLES = 5
MOROTTI_SKIP_FOR = {"APD30", "APD50", "APD60", "APD90", "Triangulation 30",
                    "Triangulation 60", "dV/dtmax steepest",
                    "Resting Membrane Potential", "AP_max", "AP_amp"}
