#!/usr/bin/env python
"""G0 go/no-go report (UAV-VisLoc sites 01 + 11, 1 km prior window).

usage: python analysis/g0_report.py results/reg/g0_uav.csv results/reg/g0_oracle.csv results/reg/g0_local.csv

Per site x mode: Median / Mean / P95 error, Success@25/50/100 m, candidate
recall (any top-k peak within 25 m of GT), buildings/query, and the same
metrics for two random baselines on the SAME queries (prior centre; uniform
point in the window). Stratified by the query's building count (a query-side
quantity, no GT). Also writes a markdown file next to the first CSV.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BINS = [0, 5, 10, 20, 40, 10**9]
LABELS = ["0-4", "5-9", "10-19", "20-39", ">=40"]


def metrics(e):
    e = np.asarray(e, float)
    e = e[np.isfinite(e)]
    if len(e) == 0:
        return dict(n=0)
    return dict(n=len(e), median=np.median(e), mean=e.mean(), p95=np.percentile(e, 95),
                S25=(e <= 25).mean(), S50=(e <= 50).mean(), S100=(e <= 100).mean())


def block(d, label):
    rows = []
    for site, g in d.groupby("site"):
        for name, col in (("registration", "err_m"), ("random: prior centre", "prior_err_m"),
                          ("random: uniform in window", "rand_err_m")):
            if col not in g:
                continue
            m = metrics(g[col]); m.update(site=site, method=name if name != "registration" else label)
            if name == "registration":
                m["cand_recall25"] = g["any_peak_within_25m"].mean() if "any_peak_within_25m" in g else np.nan
                m["buildings_med"] = g["uav_n_buildings"].median() if "uav_n_buildings" in g else g["n_buildings"].median()
            rows.append(m)
    return rows


def main():
    files = sys.argv[1:]
    allrows, strat = [], []
    for f in files:
        d = pd.read_csv(f)
        d["site"] = d["site"].astype(str).str.zfill(2)
        label = d["mode"].iloc[0] if "mode" in d else Path(f).stem
        allrows += block(d, label)
        nb = d["uav_n_buildings"] if "uav_n_buildings" in d else d["n_buildings"]
        d["nb_bin"] = pd.cut(nb, BINS, right=False, labels=LABELS)
        for (site, b), g in d.groupby(["site", "nb_bin"], observed=True):
            m = metrics(g["err_m"]); m.update(site=site, mode=label, buildings=b,
                                             rand_S50=(g["rand_err_m"] <= 50).mean() if "rand_err_m" in g else np.nan)
            strat.append(m)
    T = pd.DataFrame(allrows)[["site", "method", "n", "median", "mean", "p95", "S25", "S50", "S100", "cand_recall25", "buildings_med"]]
    S = pd.DataFrame(strat)[["site", "mode", "buildings", "n", "median", "p95", "S25", "S50", "S100", "rand_S50"]]
    pd.set_option("display.width", 200)
    print(T.round(3).to_string(index=False)); print(); print(S.round(3).to_string(index=False))
    out = Path(files[0]).with_name("g0_report.md")
    out.write_text("# G0 report\n\n" + T.round(3).to_markdown(index=False) + "\n\n## By building count\n\n"
                   + S.round(3).to_markdown(index=False) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
