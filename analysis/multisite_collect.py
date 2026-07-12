#!/usr/bin/env python
"""Assemble per-site + aggregate metrics for the multi-site OURS benchmark into
outputs/paper/multisite_metrics.json. Reads:
  - outputs/eval/<site>/records_main.csv     (query-stage per-image results)
  - logs/diag_multisite.log                  (phase_a_diag: coverage/retrieval/L1/Hough)
  - logs/pf_multisite.log                    (pf_pipeline: frame support + PF error)
Robust to missing sites/files. CPU-only, no torch.
"""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np
import pandas as pd

SITES = ["01", "02", "03", "08", "09", "11"]
TERRAIN = {"01": "riverside-urban", "02": "coastal-agri", "03": "hilly-veg",
           "08": "built-up", "09": "built-up", "11": "arid-town"}
EVAL = Path("outputs/eval")
FIGDIR = Path("outputs/paper"); FIGDIR.mkdir(parents=True, exist_ok=True)


def f(x):
    try: return float(x)
    except Exception: return float("nan")


def records_metrics(site):
    p = EVAL / site / "records_main.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df = df[df["gt_px_x"].notna()].copy()
    n = len(df)
    if n == 0:
        return None
    err = pd.to_numeric(df["error_m"], errors="coerce").dropna().values
    rerr = pd.to_numeric(df.get("refined_error_m", pd.Series([])), errors="coerce").dropna().values
    votes = pd.to_numeric(df.get("gt_patch_votes", pd.Series([0]*n)), errors="coerce").fillna(0).values
    rank = pd.to_numeric(df.get("gt_patch_rank", pd.Series([])), errors="coerce").dropna().values
    def rec(a, t): return round(100.0*np.mean(a < t), 2) if len(a) else None
    # CFBVM / UAV-VisLoc-style meter-level accuracy ladder (MA@K)
    ladder = {f"MA@{t}m_pct": rec(err, t) for t in (5, 10, 20, 30, 50, 100, 200, 500)}
    return {
        "n_queries": int(n),
        "median_err_m": round(float(np.median(err)), 1) if len(err) else None,
        "mean_err_m": round(float(np.mean(err)), 1) if len(err) else None,
        "median_refined_err_m": round(float(np.median(rerr)), 1) if len(rerr) else None,
        **ladder,
        "gt_gets_vote_pct": round(100.0*np.mean(votes > 0), 1),
        "median_gt_patch_rank": round(float(np.median(rank)), 1) if len(rank) else None,
    }


def split_sections(text, marker_re):
    """Yield (site, block) splitting log text on a per-site marker regex."""
    idxs = [(m.group(1), m.start()) for m in re.finditer(marker_re, text)]
    for i, (site, start) in enumerate(idxs):
        end = idxs[i+1][1] if i+1 < len(idxs) else len(text)
        yield site, text[start:end]


def diag_metrics():
    p = Path("logs/diag_multisite.log")
    out = {}
    if not p.exists():
        return out
    txt = p.read_text(errors="replace")
    for site, blk in split_sections(txt, r"===== site (\d+)"):
        d = {}
        m = re.search(r"coverage .*?: ([\d.]+)%", blk);            d["coverage_pct"] = f(m.group(1)) if m else None
        m = re.search(r"true-match descriptor L1: median=([\d.]+)", blk); d["true_match_L1_median"] = f(m.group(1)) if m else None
        m = re.search(r"retrieved in top-(\d+): ([\d.]+)%", blk)
        if m: d["retrieval_topk_K"] = int(m.group(1)); d["retrieval_topk_pct"] = f(m.group(2))
        m = re.search(r"Hough position error \(m\): mean=([\d.]+) median=([\d.]+)", blk)
        if m: d["hough_mean_m"] = f(m.group(1)); d["hough_median_m"] = f(m.group(2))
        out[site] = d
    return out


def pf_metrics():
    p = Path("logs/pf_multisite.log")
    out = {}
    if not p.exists():
        return out
    txt = p.read_text(errors="replace")
    for site, blk in split_sections(txt, r"===== PF site (\d+)"):
        d = {}
        m = re.search(r"frames with >=1 vote within .*?: ([\d.]+)%", blk); d["frames_gt_supported_pct"] = f(m.group(1)) if m else None
        m = re.search(r"GT-support votes/frame: median=([\d.]+)", blk);    d["median_support"] = f(m.group(1)) if m else None
        m = re.search(r"error over last \d+ frames: median=([\d.]+)m", blk); d["pf_median_err_m"] = f(m.group(1)) if m else None
        d["pf_converged"] = ("never converged" not in blk) and ("converged" in blk)
        out[site] = d
    return out


def main():
    diag = diag_metrics(); pf = pf_metrics()
    sites = {}
    for s in SITES:
        rec = records_metrics(s)
        if rec is None and s not in diag and s not in pf:
            continue
        sites[s] = {"terrain": TERRAIN.get(s, "?"), "records": rec,
                    "diag": diag.get(s, {}), "pf": pf.get(s, {})}
    # cross-site aggregate over available records
    med_errs = [v["records"]["median_err_m"] for v in sites.values()
                if v["records"] and v["records"]["median_err_m"] is not None]
    ret = [v["diag"].get("retrieval_topk_pct") for v in sites.values() if v["diag"].get("retrieval_topk_pct") is not None]
    cov = [v["diag"].get("coverage_pct") for v in sites.values() if v["diag"].get("coverage_pct") is not None]
    agg = {
        "n_sites": len(sites),
        "median_err_m_range": [min(med_errs), max(med_errs)] if med_errs else None,
        "retrieval_topk_pct_max": max(ret) if ret else None,
        "coverage_pct_range": [min(cov), max(cov)] if cov else None,
    }
    obj = {"sites": sites, "aggregate": agg}
    (FIGDIR / "multisite_metrics.json").write_text(json.dumps(obj, indent=2))
    print(json.dumps(obj, indent=2))
    print(f"\n[collect] wrote {FIGDIR/'multisite_metrics.json'} for {len(sites)} sites", flush=True)


if __name__ == "__main__":
    main()
