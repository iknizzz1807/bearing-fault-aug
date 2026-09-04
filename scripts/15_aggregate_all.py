#!/usr/bin/env python3
"""Gộp mean±std recall (3 seed) cho mọi nhaành từ results/*.json và so với số trong REPORT.

Đọc ONly từ JSON (không số tay). Chạy:
  python scripts/15_aggregate_all.py
"""
import json, glob, statistics, sys, re
from collections import defaultdict

BASE = "results"
SEEDS = ["s21", "s42", "s97"]
KS = [10, 20, 50]

# family -> (glob_prefix, key cho 'after', key cho 'before', n_synth label)
# fix_* : 'before'/'after', incip : 'baseline'/'interp_align_amp'
FAMILIES = {
    "interp_align": ("fix_interp_align", "after", "before"),
    "physics":      ("fix_physics",      "after", "before"),
    "amp_align":    ("fx_amp_align",     "after", "before"),
    "interpolate":  ("fx_optimized",     "after", "before"),
    "ddpm":         ("fx_ddpm",          "after", "before"),
    "heuristic":    ("fx_heuristic",     "after", "before"),
    "aligned":      ("fx_aligned",       "after", "before"),
    "incipient":    ("incip",            "interp_align_amp", "baseline"),
}


def load(fam, k):
    """Trả về list recall 'after' và 'before' (chỉ .recall) qua 3 seed."""
    prefix, after_key, before_key = FAMILIES[fam]
    afters, befores = [], []
    for s in SEEDS:
        p = f"{BASE}/{prefix}_K{k}_{s}.json"
        try:
            d = json.load(open(p))
        except FileNotFoundError:
            return None
        row = d["rows"][0]
        afters.append(row[after_key]["recall"])
        befores.append(row[before_key]["recall"])
    return afters, befores


def agg(vals):
    m = statistics.mean(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return m, sd


def main():
    print(f"{'family':<12} {'K':<4} {'mean_after':<11} {'std':<8} {'mean_before':<12} {'delta':<9}")
    table = defaultdict(dict)          # family -> k -> (after_mean, before_mean)
    incip_table = defaultdict(dict)    # family -> k -> (after_mean, before_mean)
    for fam in FAMILIES:
        for k in KS:
            res = load(fam, k)
            if res is None:
                print(f"{fam:<12} K={k}  (missing files)")
                continue
            afters, befores = res
            am, asd = agg(afters)
            bm, bsd = agg(befores)
            # delta theo từng seed rồi mean
            deltas = [a - b for a, b in zip(afters, befores)]
            dmean, dsd = agg(deltas)
            table[fam][k] = (am, bm)
            if fam == "incipient":
                incip_table[fam][k] = (am, bm)
            print(f"{fam:<12} {k:<4} {am:<11.3f} {asd:<8.3f} {bm:<12.3f} {dmean:<9.3f}")
    # Lưu
    json.dump(
        {fam: {str(k): {"after": t[0], "before": t[1]} for k, t in ks.items()} for fam, ks in table.items()},
        open(f"{BASE}/aggregated_recall.json", "w"), indent=1,
    )
    print("\nSaved -> results/aggregated_recall.json")


if __name__ == "__main__":
    main()
