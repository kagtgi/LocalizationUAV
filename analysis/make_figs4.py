#!/usr/bin/env python
"""Area-level Ekeland figure: per-site retrieval@top-50 for four descriptors
(plain density+layout, re-implemented BRM, re-implemented CFBVM concentric
rings, plain+Ekeland-angle histograms). Matches tab:area exactly. Numbers are
the verified, already-committed results (area_ekeland_d4-equivalent from
area_ekeland_probe2.py and baselines_probe.py, --sites 01 02 03 08 11 --limit
100); hardcoded here since the D_max sweep renamed the live JSON outputs.
CPU-only, no data dependency at figure-generation time.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gstyle
gstyle.setup()

FIGDIR = Path("outputs/paper")
TERRAIN = {"01": "urban", "02": "coastal", "03": "hilly", "08": "built-up", "11": "arid"}
ORDER = ["01", "02", "03", "08", "11"]

# top-50 coarse recall (%), verified + committed to paper_draft/main.tex tab:area
DATA = {
    "plain":       {"01": 5.5, "02": 2.0, "03": 3.2, "08": 4.3, "11": 0.0},
    "brm":         {"01": 0.0, "02": 2.0, "03": 1.6, "08": 0.0, "11": 3.4},
    "cfbvm_rings": {"01": 3.6, "02": 6.1, "03": 0.0, "08": 2.2, "11": 5.1},
    "ekeland":     {"01": 9.1, "02": 10.2, "03": 6.4, "08": 2.2, "11": 1.7},
}
CHANCE = 3.7  # top-50 / ~1350 avg cells


def main():
    keys = ORDER
    labels = [f"{s}\n{TERRAIN[s]}" for s in keys]
    modes = ["plain", "brm", "cfbvm_rings", "ekeland"]
    mode_labels = ["Plain (density+layout)", "BRM (re-impl.)", "CFBVM-rings (re-impl.)", "+ Ekeland (ours)"]
    colors = [gstyle.INK_MUTED, gstyle.YELLOW, gstyle.VIOLET, gstyle.BLUE]

    fig, ax = plt.subplots(figsize=(7.6, 3.5))
    fig.subplots_adjust(bottom=0.16, top=0.80, left=0.09, right=0.98)
    gstyle.clean_axes(ax, grid_axis="y")
    x = np.arange(len(keys))
    n = len(modes)
    w = 0.82 / n
    for i, (mode, lab, col) in enumerate(zip(modes, mode_labels, colors)):
        vals = [DATA[mode][s] for s in keys]
        offset = (i - (n - 1) / 2) * w
        bars = ax.bar(x + offset, vals, w * 0.92, color=col, zorder=3, label=lab)
        if mode == "ekeland":
            for b, v in zip(bars, vals):
                gstyle.annotate(ax, b.get_x() + b.get_width() / 2, v, f"{v:.1f}",
                                color=gstyle.INK_SECONDARY, fontsize=7.6, weight="medium")

    ax.axhline(CHANCE, ls="-", lw=0.9, color=gstyle.CRITICAL, zorder=2)
    ax.annotate("chance", (len(keys) - 0.5, CHANCE + 0.25), fontsize=7.8,
                color=gstyle.CRITICAL, ha="right", va="bottom")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.6)
    ax.set_ylabel("coarse-localization recall @top-50 (%)")
    ax.set_ylim(0, 11.8)
    ax.set_title("Area-level: Ekeland beats plain (p=0.02) and BRM (p=0.01) pooled;\n"
                 "directionally ahead of CFBVM-rings (p=0.23) -- best on 3 of 5 sites",
                 fontsize=9.6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.30), ncol=4, fontsize=7.6,
              handlelength=1.3, columnspacing=1.1, frameon=False)
    fig.savefig(FIGDIR / "fig_area_ekeland.pdf")
    fig.savefig(FIGDIR / "fig_area_ekeland.png", dpi=300)
    print("[fig] fig_area_ekeland written", flush=True)


if __name__ == "__main__":
    main()
