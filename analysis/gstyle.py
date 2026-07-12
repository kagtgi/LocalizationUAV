"""Shared 'Google Research figure' style: Google Sans typography, a validated
CVD-safe categorical palette, thin marks, hairline recessive grid/axes,
generous padding. Import and call setup() before creating any figure."""
import glob
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# --------------------------------------------------------------------------- palette (validated, light mode)
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"

BLUE = "#2a78d6"
AQUA = "#1baf7a"
YELLOW = "#eda100"
GREEN = "#008300"
VIOLET = "#4a3aa7"
RED = "#e34948"
MAGENTA = "#e87ba4"
ORANGE = "#eb6834"
CAT = [BLUE, AQUA, YELLOW, GREEN, VIOLET, RED, MAGENTA, ORANGE]

GOOD = "#0ca30c"
CRITICAL = "#d03b3b"


def setup():
    for f in glob.glob("/home/vien_toan/.fonts/*.ttf"):
        fm.fontManager.addfont(f)
    matplotlib.rcParams.update({
        "font.family": "Google Sans",
        "font.size": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.12,
        "text.color": INK,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "axes.titlesize": 10.5,
        "axes.titleweight": "medium",
        "axes.labelsize": 9.5,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.axisbelow": True,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelsize": 8.8,
        "ytick.labelsize": 8.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        # frameon=True + a plain white, edge-less patch: invisible against a
        # white axes background, but essential wherever a legend sits over an
        # image or dense plot (fig_votecloud, fig_top100_demo) -- prevents
        # legend text silently overlapping the content beneath it.
        "legend.frameon": True,
        "legend.facecolor": SURFACE,
        "legend.edgecolor": "none",
        "legend.framealpha": 0.92,
        "legend.fontsize": 8.8,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
    })


def clean_axes(ax, grid_axis=None):
    """Hairline recessive grid (optional), thin axis, no top/right spine."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)


def annotate(ax, x, y, text, color=INK_SECONDARY, fontsize=8, ha="center", va="bottom", weight="normal"):
    ax.annotate(text, (x, y), ha=ha, va=va, fontsize=fontsize, color=color, fontweight=weight)
