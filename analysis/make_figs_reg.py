#!/usr/bin/env python
"""Data figures for the StructReg journal paper (numbers come only from CSVs).

  fig_g0_cdf      error CDF on the G0 sites: StructReg vs oracle structure vs
                  random (prior centre / uniform in window) vs M0-in-window
  fig_g0_strata   Success@50 m by query building count (query-side strata)

usage: python analysis/make_figs_reg.py results/reg paper_journal/figs
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gstyle as S  # noqa: E402

BINS = [0, 5, 10, 20, 40, 10**9]
LABELS = ["0–4", "5–9", "10–19", "20–39", "≥40"]


def cdf(ax, e, label, color, ls="-"):
    e = np.sort(np.asarray(e, float)[np.isfinite(e)])
    if len(e) == 0:
        return
    ax.plot(e, np.arange(1, len(e) + 1) / len(e), ls, color=color, lw=1.6, label=label)


def fig_g0(res: Path, out: Path):
    uav = pd.read_csv(res / "g0_uav.csv")
    orc = pd.read_csv(res / "g0_oracle.csv") if (res / "g0_oracle.csv").exists() else None
    m0 = None
    for f in sorted(Path("outputs/m0").glob("**/records_main.csv")):
        d = pd.read_csv(f)
        if "m0win_err_m" in d:
            m0 = d if m0 is None else pd.concat([m0, d])
    sites = sorted(uav["site"].astype(str).str.zfill(2).unique())
    fig, axes = plt.subplots(1, len(sites), figsize=(7.2, 2.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, s in zip(axes, sites):
        u = uav[uav["site"].astype(str).str.zfill(2) == s]
        cdf(ax, u["err_m"], "StructReg", S.BLUE)
        if orc is not None:
            o = orc[orc["site"].astype(str).str.zfill(2) == s]
            cdf(ax, o["err_m"], "oracle structure", S.AQUA, "--")
        if m0 is not None:
            mm = m0[m0["site"].astype(str).str.zfill(2) == s]
            cdf(ax, mm["m0win_err_m"], "M0 (window)", S.YELLOW)
        cdf(ax, u["rand_err_m"], "uniform in window", S.INK_MUTED, ":")
        cdf(ax, u["prior_err_m"], "prior centre", S.INK_MUTED, "-.")
        ax.set_xscale("log"); ax.set_xlim(1, 1500)
        ax.set_title(f"Site {s}"); ax.set_xlabel("horizontal error (m)")
        for t in (25, 50, 100):
            ax.axvline(t, color=S.GRID, lw=0.8, zorder=0)
    axes[0].set_ylabel("fraction of queries")
    axes[-1].legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.savefig(out / "fig_g0_cdf.pdf"); plt.close(fig)

    # strata
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    width = 0.8 / max(len(sites), 1)
    for i, s in enumerate(sites):
        u = uav[uav["site"].astype(str).str.zfill(2) == s].copy()
        nb = u["uav_n_buildings"] if "uav_n_buildings" in u else u["n_buildings"]
        u["bin"] = pd.cut(nb, BINS, right=False, labels=LABELS)
        g = u.groupby("bin", observed=False)
        v = g["err_m"].apply(lambda e: np.mean(e <= 50) if len(e) else np.nan)
        n = g.size()
        x = np.arange(len(LABELS)) + (i - (len(sites) - 1) / 2) * width
        ax.bar(x, v.values, width, color=S.CAT[i], label=f"site {s}")
        for xi, vi, ni in zip(x, v.values, n.values):
            if ni:
                ax.text(xi, (0 if np.isnan(vi) else vi) + 0.02, str(ni), ha="center", fontsize=6.5, color=S.INK_MUTED)
    ax.set_xticks(np.arange(len(LABELS))); ax.set_xticklabels(LABELS)
    ax.set_xlabel("buildings observed in the query"); ax.set_ylabel("Success@50 m")
    ax.set_ylim(0, 1.05); ax.legend(frameon=False, fontsize=7.5)
    fig.savefig(out / "fig_g0_strata.pdf"); plt.close(fig)


def main():
    res = Path(sys.argv[1] if len(sys.argv) > 1 else "results/reg")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "paper_journal/figs")
    out.mkdir(parents=True, exist_ok=True)
    S.setup()
    fig_g0(res, out)
    print("figures ->", out)


if __name__ == "__main__":
    main()
