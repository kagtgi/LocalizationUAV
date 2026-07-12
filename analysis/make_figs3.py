#!/usr/bin/env python
"""Cross-site figure for the multi-site OURS benchmark. Reads
outputs/paper/multisite_metrics.json and produces fig_multisite.{pdf,png}:
two single-axis panels (median position error; meter-level accuracy @500m)
side by side -- median error and MA@500m stay in the same ballpark on every
terrain, and the deliberately separate axes avoid the dual-axis anti-pattern
(two arbitrarily-aligned y-scales inventing a correlation that isn't there).
CPU-only.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gstyle
gstyle.setup()

FIGDIR = Path("outputs/paper")
C_ERR = gstyle.RED
C_MA = gstyle.BLUE

ORDER = ["01", "02", "03", "08", "11"]
SHORT_TERRAIN = {"01": "urban", "02": "coastal", "03": "hilly", "08": "built-up", "11": "arid"}


def main():
    obj = json.loads((FIGDIR / "multisite_metrics.json").read_text())
    sites = obj["sites"]
    keys = [s for s in ORDER if s in sites and sites[s].get("records")]
    labels = [f"{s}\n{SHORT_TERRAIN[s]}" for s in keys]
    err = [sites[s]["records"].get("median_err_m", np.nan) for s in keys]
    ma500 = [sites[s]["records"].get("MA@500m_pct", np.nan) for s in keys]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.3))
    fig.subplots_adjust(wspace=0.38, bottom=0.24, top=0.85, left=0.10, right=0.97)
    x = np.arange(len(keys))

    gstyle.clean_axes(ax, grid_axis="y")
    bars = ax.bar(x, err, width=0.6, color=C_ERR, zorder=3)
    ax.set_yscale("log")
    ax.set_ylabel("median position error (m, log)")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.2)
    ax.set_ylim(20, 9000)
    ax.set_xlim(-0.75, len(keys) - 0.25)
    ax.axhline(30, ls="-", lw=0.9, color=gstyle.INK_MUTED, zorder=2)
    ax.annotate("30 m target", (-0.42, 34), fontsize=7.6, color=gstyle.INK_MUTED,
                ha="left", va="bottom", zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    for b, e in zip(bars, err):
        if e == e:
            gstyle.annotate(ax, b.get_x() + b.get_width() / 2, e, f"{e:.0f}",
                            color=gstyle.INK_SECONDARY, fontsize=8, weight="medium")
    ax.set_title("(a) median error stays km-scale\non every terrain", fontsize=9.8)

    gstyle.clean_axes(ax2, grid_axis="y")
    bars2 = ax2.bar(x, ma500, width=0.6, color=C_MA, zorder=3)
    ax2.set_ylabel("meter-level accuracy @500m (%)")
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=8.2)
    ax2.set_ylim(0, 4.2)
    for b, r in zip(bars2, ma500):
        if r == r:
            gstyle.annotate(ax2, b.get_x() + b.get_width() / 2, r, f"{r:.1f}%",
                            color=gstyle.INK_SECONDARY, fontsize=8, weight="medium")
    ax2.set_title("(b) MA@500m never\nexceeds 3.1%", fontsize=9.8)

    fig.suptitle("Cross-site: single-frame localization fails on every terrain", fontsize=10.5, y=0.99)
    fig.savefig(FIGDIR / "fig_multisite.pdf")
    fig.savefig(FIGDIR / "fig_multisite.png", dpi=300)
    print("[fig] fig_multisite written", flush=True)


if __name__ == "__main__":
    main()
