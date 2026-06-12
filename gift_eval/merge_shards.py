#!/usr/bin/env python3
"""Concatenate shard CSVs produced by `run_eval.py --num-shards N --shard K`,
sort, dedupe by dataset, write the merged CSV, and recompute LB MASE.

Usage::
    python gift_eval/merge_shards.py 'gift_eval/all_results_shard*of4.csv' gift_eval/all_results.csv
"""
from __future__ import annotations

import argparse
import glob
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gmean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUBMIT_DIR = Path(__file__).resolve().parent
# Vendored seasonal-naive baseline (LB MASE denominator), shipped in gift_eval/data/.
SN_CSV = str(SUBMIT_DIR / "data" / "seasonal_naive_all_results.csv")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("merge_shards")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pattern", help="Glob for shard CSVs (e.g. 'gift_eval/all_results_shard*of4.csv').")
    ap.add_argument("out_csv", help="Output merged CSV path.")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.pattern))
    if not paths:
        log.error("No files match: %s", args.pattern)
        sys.exit(1)
    log.info("Merging %d shard CSVs ...", len(paths))
    for p in paths:
        log.info("  - %s", p)

    dfs = [pd.read_csv(p) for p in paths]
    merged = pd.concat(dfs, ignore_index=True)
    n_before = len(merged)
    merged = merged.drop_duplicates(subset="dataset", keep="last")
    n_after = len(merged)
    if n_before != n_after:
        log.warning("Dropped %d duplicate (dataset, ...) rows", n_before - n_after)
    merged = merged.sort_values(by="dataset").reset_index(drop=True)
    merged.to_csv(args.out_csv, index=False)
    log.info("Wrote %s (%d rows)", args.out_csv, len(merged))

    try:
        sn = pd.read_csv(SN_CSV).drop_duplicates(subset="dataset", keep="last")
        sn_map = sn.set_index("dataset")["eval_metrics/MASE[0.5]"]
        paired = (merged.set_index("dataset")["eval_metrics/MASE[0.5]"].astype(float)
                  / sn_map).replace([np.inf, -np.inf], np.nan).dropna()
        lb = float(gmean(paired.values))
        (SUBMIT_DIR / "lb_mase.txt").write_text(f"{lb:.4f}\n")
        log.info("LB MASE = %.4f   (deployed = 0.6746)", lb)
        log.info("  paired = %d / %d datasets vs seasonal_naive", len(paired), len(merged))
    except Exception as e:
        log.warning("Could not compute LB MASE: %s", e)


if __name__ == "__main__":
    main()
