"""
Shared plotting style and palette for every chart in the project.

The palette is not arbitrary. The categorical colours were checked
programmatically for lightness band, chroma floor, colour-vision-deficiency
separation and contrast against the chart surface, rather than picked by eye.

The rule that drives every colour choice here:
  - encoding a MAGNITUDE ("how much") -> one hue, light to dark  (sequential)
  - encoding an IDENTITY  ("which one") -> distinct hues in a fixed order
    (categorical), never cycled
"""

import sys

import matplotlib

# Backend selection depends on the environment:
# - From a terminal there is no display, and plt.show() would *block* the script
#   until a window is closed. Hence Agg, which only writes files.
# - Under Jupyter the ipykernel module is already loaded and sets up its own
#   inline backend; forcing Agg there would mean no chart ever renders in the
#   notebook.
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ---- Surfaces and ink ----
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"

# ---- Categorical palette (fixed order, never reassigned per chart) ----
SERIES = ["#2a78d6", "#eb6834"]   # slot 1: blue, slot 2: orange

# ---- Single-hue sequential ramp (blue), light -> dark ----
_BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
              "#256abf", "#184f95", "#0d366b"]
BLUE_SEQ = LinearSegmentedColormap.from_list("blue_seq", _BLUE_RAMP)

# The single hue for magnitude bars, where colour carries no identity because
# the axis labels already do.
BAR_COLOR = "#2a78d6"


def apply_style() -> None:
    """Applies a consistent style: recessive axes and grid, no chart junk."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.titlecolor": TEXT_PRIMARY,
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.labelsize": 10,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2,
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def despine(ax, keep=("left", "bottom")) -> None:
    """Removes the chart borders that carry no information."""
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)


def save(fig, path: str) -> None:
    """Saves a figure and closes it, so a long run does not accumulate memory."""
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved: {path}")
