#!/usr/bin/env python
"""Statistical rigor pass for the area-level Ekeland-vs-plain result (review
point 3: "no confidence intervals, no significance test... not acceptable at
Q1"). Reads outputs/eval/<site>/area_per_frame.json (per-frame paired hit
indicators dumped by area_ekeland_probe.py --dump-per-frame) and computes:
  - per-site bootstrap 95% CI on the recall delta (ekeland - plain)
  - per-site paired permutation test (H0: no systematic difference)
  - POOLED (all sites concatenated) bootstrap CI + permutation test - the
    headline number, since individual per-site n~70-90 gives wide per-site CIs
No numpy RNG seed-dependent surprises: uses a fixed seed for reproducibility.
Self-test (--selftest): synthetic null (delta=0) must give p roughly uniform /
CI straddling zero; synthetic large effect must give p<0.05 and CI excluding 0.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def bootstrap_ci(plain, ekeland, n_boot=10000, seed=0, alpha=0.05):
    """plain, ekeland: (n,) boolean/float paired arrays (same frames, same order)."""
    rng = np.random.default_rng(seed)
    n = len(plain)
    obs_delta = float(np.mean(ekeland) - np.mean(plain))
    idx = rng.integers(0, n, size=(n_boot, n))
    deltas = ekeland[idx].mean(axis=1) - plain[idx].mean(axis=1)
    lo, hi = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return obs_delta, float(lo), float(hi)


def paired_permutation_test(plain, ekeland, n_perm=10000, seed=0):
    """H0: no systematic difference between paired plain/ekeland outcomes.
    Randomly flips which member of each pair is 'plain' vs 'ekeland' (valid
    under H0 of exchangeability) and rebuilds the null distribution of the
    mean delta."""
    rng = np.random.default_rng(seed)
    n = len(plain)
    obs_delta = float(np.mean(ekeland) - np.mean(plain))
    stacked = np.stack([plain, ekeland], axis=1)  # (n,2)
    flips = rng.integers(0, 2, size=(n_perm, n))
    null_deltas = np.empty(n_perm)
    for i in range(n_perm):
        a = np.where(flips[i] == 0, stacked[:, 0], stacked[:, 1])
        b = np.where(flips[i] == 0, stacked[:, 1], stacked[:, 0])
        null_deltas[i] = b.mean() - a.mean()
    p = float(np.mean(np.abs(null_deltas) >= abs(obs_delta)))
    return obs_delta, p


def mcnemar(plain, ekeland):
    """2x2 discordant-pair test for paired binary outcomes."""
    plain = np.asarray(plain, bool); ekeland = np.asarray(ekeland, bool)
    b = int(np.sum(plain & ~ekeland))   # plain hit, ekeland miss
    c = int(np.sum(~plain & ekeland))   # plain miss, ekeland hit
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "chi2": 0.0, "p_approx": 1.0}
    chi2 = (abs(b - c) - 1) ** 2 / n  # continuity-corrected
    # crude chi2(1) survival via complementary error function relation
    from math import erfc, sqrt
    p_approx = erfc(sqrt(chi2 / 2))
    return {"b": b, "c": c, "chi2": round(chi2, 3), "p_approx": round(p_approx, 4)}


def selftest():
    rng = np.random.default_rng(0)
    print("[selftest] null case (no true effect): delta CI should straddle 0, p should be large", flush=True)
    n_fail = 0
    for _ in range(20):
        n = 80
        plain = rng.binomial(1, 0.10, n).astype(float)
        ekeland = rng.binomial(1, 0.10, n).astype(float)  # same true rate
        d, lo, hi = bootstrap_ci(plain, ekeland, n_boot=2000, seed=int(rng.integers(1e6)))
        _, p = paired_permutation_test(plain, ekeland, n_perm=2000, seed=int(rng.integers(1e6)))
        if not (lo < 0 < hi) and p < 0.05:
            n_fail += 1  # both CI excludes 0 AND p<0.05 on a true null -> bad
    print(f"  false-positive rate over 20 null trials: {n_fail}/20 (expect low, <=2 by chance at alpha=0.05*2)", flush=True)

    print("[selftest] large true effect: delta CI should exclude 0, p should be small", flush=True)
    ok = 0
    for _ in range(20):
        n = 80
        plain = rng.binomial(1, 0.05, n).astype(float)
        ekeland = rng.binomial(1, 0.40, n).astype(float)  # large true effect
        d, lo, hi = bootstrap_ci(plain, ekeland, n_boot=2000, seed=int(rng.integers(1e6)))
        _, p = paired_permutation_test(plain, ekeland, n_perm=2000, seed=int(rng.integers(1e6)))
        if lo > 0 and p < 0.05:
            ok += 1
    print(f"  true-positive rate over 20 large-effect trials: {ok}/20 (expect high)", flush=True)


def load_site(site_dir, metric_key, filename="area_per_frame.json", mode_a="plain", mode_b="ekeland"):
    pf_path = Path(site_dir) / filename
    if not pf_path.exists():
        return None
    obj = json.loads(pf_path.read_text())
    a_rows = {r["name"]: r for r in obj.get(mode_a, [])}
    b_rows = {r["name"]: r for r in obj.get(mode_b, [])}
    names = sorted(set(a_rows) & set(b_rows))
    if not names:
        return None
    a = np.array([float(a_rows[n][metric_key]) for n in names])
    b = np.array([float(b_rows[n][metric_key]) for n in names])
    return a, b, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/eval")
    ap.add_argument("--sites", nargs="+", default=["01", "02", "03", "11"])
    ap.add_argument("--metric", default="hit_top50", help="per-frame field name to test")
    ap.add_argument("--n-boot", type=int, default=10000, dest="n_boot")
    ap.add_argument("--n-perm", type=int, default=10000, dest="n_perm")
    ap.add_argument("--file", default="area_per_frame.json", help="per-frame dump filename")
    ap.add_argument("--mode-a", default="plain", dest="mode_a")
    ap.add_argument("--mode-b", default="ekeland", dest="mode_b")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    result = {"per_site": {}}
    all_plain, all_ekeland = [], []
    for s in args.sites:
        d = load_site(Path(args.out) / s, args.metric, args.file, args.mode_a, args.mode_b)
        if d is None:
            print(f"[skip] {s}: no area_per_frame.json (rerun area_ekeland_probe.py --dump-per-frame)")
            continue
        plain, ekeland, names = d
        obs_delta, lo, hi = bootstrap_ci(plain, ekeland, args.n_boot)
        _, p_perm = paired_permutation_test(plain, ekeland, args.n_perm)
        mc = mcnemar(plain, ekeland)
        sig = "significant" if (lo > 0 or hi < 0) else "NOT significant"
        print(f"[{s}] n={len(names)}  plain={plain.mean()*100:.1f}%  ekeland={ekeland.mean()*100:.1f}%  "
              f"delta={obs_delta*100:+.1f}pp  95%CI=[{lo*100:+.1f},{hi*100:+.1f}]pp  "
              f"perm_p={p_perm:.4f}  McNemar b={mc['b']} c={mc['c']} p~={mc['p_approx']:.4f}  -> {sig}", flush=True)
        result["per_site"][s] = {
            "n": len(names), "plain_pct": round(float(plain.mean() * 100), 1),
            "ekeland_pct": round(float(ekeland.mean() * 100), 1),
            "delta_pp": round(float(obs_delta * 100), 1),
            "ci95_pp": [round(lo * 100, 1), round(hi * 100, 1)],
            "perm_p": round(p_perm, 4), "mcnemar": mc, "significant": bool(lo > 0 or hi < 0),
        }
        all_plain.append(plain); all_ekeland.append(ekeland)

    if all_plain:
        pooled_plain = np.concatenate(all_plain)
        pooled_ekeland = np.concatenate(all_ekeland)
        obs_delta, lo, hi = bootstrap_ci(pooled_plain, pooled_ekeland, args.n_boot)
        _, p_perm = paired_permutation_test(pooled_plain, pooled_ekeland, args.n_perm)
        mc = mcnemar(pooled_plain, pooled_ekeland)
        sig = "significant" if (lo > 0 or hi < 0) else "NOT significant"
        print(f"\n[POOLED n={len(pooled_plain)}] plain={pooled_plain.mean()*100:.1f}%  "
              f"ekeland={pooled_ekeland.mean()*100:.1f}%  delta={obs_delta*100:+.1f}pp  "
              f"95%CI=[{lo*100:+.1f},{hi*100:+.1f}]pp  perm_p={p_perm:.4f}  "
              f"McNemar b={mc['b']} c={mc['c']} p~={mc['p_approx']:.4f}  -> {sig}", flush=True)
        result["pooled"] = {
            "n": int(len(pooled_plain)), "plain_pct": round(float(pooled_plain.mean() * 100), 1),
            "ekeland_pct": round(float(pooled_ekeland.mean() * 100), 1),
            "delta_pp": round(float(obs_delta * 100), 1),
            "ci95_pp": [round(lo * 100, 1), round(hi * 100, 1)],
            "perm_p": round(p_perm, 4), "mcnemar": mc, "significant": bool(lo > 0 or hi < 0),
        }

    Path("outputs/paper").mkdir(parents=True, exist_ok=True)
    Path(f"outputs/paper/stats_pass_{args.metric}.json").write_text(json.dumps(result, indent=2))
    print(f"\n[stats] wrote outputs/paper/stats_pass_{args.metric}.json", flush=True)


if __name__ == "__main__":
    main()
