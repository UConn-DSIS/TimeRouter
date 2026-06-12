"""Evaluate TimeRouter on every GIFT-EVAL test config and write all_results.csv.

This is the GIFT-EVAL submission entry point. It mirrors the structure of
https://github.com/SalesforceAIResearch/gift-eval/blob/main/notebooks/chronos-2.ipynb
exactly:

    for ds_name in ALL_DATASETS:
        for term in {short, medium, long}:
            ds = Dataset(name=ds_name, term=term, ...)
            predictor = TimeRouterPredictor(prediction_length=ds.prediction_length, ...)
            for window_idx in range(ds.test_data.windows):
                inputs = islice(ds.test_data.input, window_idx, None, n_windows)
                forecasts.extend(predictor.predict(inputs))
            metrics = evaluate_forecasts(forecasts, ds.test_data, metrics=METRICS, ...)

The 11-metric list, dataset-properties JSON loading, and rolling-window
prediction loop all match the official notebook.

Usage::

    python gift_eval/run_eval.py [--datasets m4_weekly]   # smoke-test on one dataset
    python gift_eval/run_eval.py                          # full 97-config run

Outputs:
    gift_eval/all_results.csv     # 97 rows + header, 15 cols, leaderboard-ready
    gift_eval/lb_mase.txt         # achieved LB MASE (deployed = 0.6746)
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import gmean

from gluonts.ev.metrics import (
    MAE, MAPE, MASE, MSE, MSIS, ND, NRMSE, RMSE, SMAPE,
    MeanWeightedSumQuantileLoss,
)
from gluonts.model import evaluate_forecasts
from gluonts.time_feature import get_seasonality

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gift_eval._data import Dataset                                            # noqa: E402

from gift_eval.predictor import QUANTILE_LEVELS, TimeRouterPredictor               # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("run_eval")

SUBMIT_DIR = Path(__file__).resolve().parent
MODEL_NAME = "TimeRouter"
# Vendored seasonal-naive baseline (LB MASE denominator), shipped in gift_eval/data/.
SN_CSV = str(SUBMIT_DIR / "data" / "seasonal_naive_all_results.csv")

# ---------------------------------------------------------------------------
#  Dataset configs (matches GIFT-EVAL notebooks)
# ---------------------------------------------------------------------------
SHORT_DATASETS = """
m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly
electricity/15T electricity/H electricity/D electricity/W
solar/10T solar/H solar/D solar/W
hospital covid_deaths
us_births/D us_births/M us_births/W
saugeenday/D saugeenday/M saugeenday/W
temperature_rain_with_missing
kdd_cup_2018_with_missing/H kdd_cup_2018_with_missing/D
car_parts_with_missing restaurant
hierarchical_sales/D hierarchical_sales/W
LOOP_SEATTLE/5T LOOP_SEATTLE/H LOOP_SEATTLE/D
SZ_TAXI/15T SZ_TAXI/H
M_DENSE/H M_DENSE/D
ett1/15T ett1/H ett1/D ett1/W ett2/15T ett2/H ett2/D ett2/W
jena_weather/10T jena_weather/H jena_weather/D
bitbrains_fast_storage/5T bitbrains_fast_storage/H
bitbrains_rnd/5T bitbrains_rnd/H
bizitobs_application bizitobs_service
bizitobs_l2c/5T bizitobs_l2c/H
""".split()

MED_LONG_DATASETS = """
electricity/15T electricity/H solar/10T solar/H
kdd_cup_2018_with_missing/H
LOOP_SEATTLE/5T LOOP_SEATTLE/H SZ_TAXI/15T M_DENSE/H
ett1/15T ett1/H ett2/15T ett2/H jena_weather/10T jena_weather/H
bitbrains_fast_storage/5T bitbrains_rnd/5T
bizitobs_application bizitobs_service
bizitobs_l2c/5T bizitobs_l2c/H
""".split()
# bitbrains_fast_storage/H and bitbrains_rnd/H are short-only on the GIFT-EVAL
# leaderboard (their /5T variants are the medium/long ones); chronos-2.ipynb
# matches this layout. Total (ds, term) configs across SHORT + MED_LONG = 97.

ALL_DATASETS = sorted(set(SHORT_DATASETS) | set(MED_LONG_DATASETS))

# Pretty dataset names (parquet name → leaderboard label).
PRETTY_NAMES = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}

# Implicit-frequency datasets (no freq in name).
IMPLICIT_FREQ = {
    "bizitobs_application": "10S", "bizitobs_service": "10S",
    "hospital": "M", "covid_deaths": "D", "car_parts_with_missing": "M",
    "restaurant": "D", "temperature_rain_with_missing": "D",
    "m4_daily": "D", "m4_hourly": "H", "m4_monthly": "M",
    "m4_quarterly": "Q", "m4_weekly": "W", "m4_yearly": "A",
}

# Standard 11-metric list from the chronos-2 notebook.
METRICS = [
    MSE(forecast_type="mean"),
    MSE(forecast_type=0.5),
    MAE(),
    MASE(),
    MAPE(),
    SMAPE(),
    MSIS(),
    RMSE(),
    NRMSE(),
    ND(),
    MeanWeightedSumQuantileLoss(quantile_levels=list(QUANTILE_LEVELS)),
]

DATASET_PROPS_PATH = str(SUBMIT_DIR / "data" / "dataset_properties.json")


def _load_dataset_props():
    try:
        return json.load(open(DATASET_PROPS_PATH))
    except FileNotFoundError:
        log.warning("dataset_properties.json not found at %s — domain/num_variates will be blank",
                    DATASET_PROPS_PATH)
        return {}


def _ds_config_key(ds_name: str, term: str) -> str:
    """Convert (ds_name, term) → the leaderboard config key '{name}/{freq}/{term}'."""
    parts = ds_name.split("/")
    if len(parts) == 1:
        ds_key, freq = parts[0], IMPLICIT_FREQ.get(parts[0], "?")
    else:
        ds_key, freq = parts[0], parts[1]
    pretty = PRETTY_NAMES.get(ds_key, ds_key).lower()
    return f"{pretty}/{freq}/{term}"


_SHARED_PREDICTOR = None  # populated lazily; reused across (ds, term) to keep
                          # tsfm_public import order stable (FlowState +
                          # PatchTSTFM share the same module name, so each new
                          # predictor instance would re-import and conflict).


def evaluate_on_dataset(predictor_kwargs: dict, ds_name: str, term: str,
                         use_multivariate_data: bool = True) -> dict:
    """Run predictor on a single (ds_name, term) and return the metric dict.

    Mirrors chronos-2.ipynb's evaluate_on_dataset: when
    ``use_multivariate_data=True`` (default), multivariate sources are kept
    as multivar (target shape (V, T)). The TimeRouterPredictor's chronos
    branch then runs in native multivariate mode; PatchTST-FM and Sundial
    are split per-variate internally.
    """
    is_multivar_source = (
        Dataset(name=ds_name, term=term, to_univariate=False).target_dim > 1
    )
    dataset = Dataset(
        name=ds_name, term=term,
        to_univariate=is_multivar_source and not use_multivariate_data,
    )
    season_length = get_seasonality(dataset.freq)

    global _SHARED_PREDICTOR
    if _SHARED_PREDICTOR is None:
        _SHARED_PREDICTOR = TimeRouterPredictor(
            prediction_length=dataset.prediction_length,
            freq=dataset.freq,
            seasonality=season_length,
            ds_name=ds_name,
            **predictor_kwargs,
        )
    else:
        _SHARED_PREDICTOR.update_for_dataset(
            ds_name=ds_name, freq=dataset.freq,
            prediction_length=dataset.prediction_length,
            seasonality=season_length,
        )
    predictor = _SHARED_PREDICTOR

    n_windows = dataset.test_data.windows
    forecast_windows = []
    for window_idx in range(n_windows):
        entries = list(itertools.islice(dataset.test_data.input,
                                          window_idx, None, n_windows))
        log.info("    window %d/%d (%d series) ...", window_idx + 1, n_windows, len(entries))
        forecast_windows.append(list(predictor.predict(entries)))

    forecasts = [item for items in zip(*forecast_windows) for item in items]
    log.info("    evaluating %d forecasts ...", len(forecasts))

    df = evaluate_forecasts(
        forecasts,
        test_data=dataset.test_data,
        metrics=METRICS,
        batch_size=1024,
        axis=None,
        mask_invalid_label=True,
        allow_nan_forecast=False,
        seasonality=season_length,
    ).reset_index(drop=True).to_dict(orient="records")
    return df[0] if df else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="Restrict to a subset of dataset names (default: all 97 configs).")
    ap.add_argument("--ckpt-dir", default=str(SUBMIT_DIR / "checkpoints_k4_chronos_flowstate_patchtst_fm_sundial"),
                    help="Local dir with seed{42..46}.json (deployed router), OR a HuggingFace "
                         "repo id like 'nkh/timerouter-v1' to download the checkpoints. If the "
                         "local default is absent, --ckpt-repo is fetched automatically.")
    ap.add_argument("--ckpt-repo", default="nkh/timerouter-v1",
                    help="HuggingFace repo to download checkpoints from when --ckpt-dir is not a "
                         "usable local directory.")
    ap.add_argument("--granite-patchtst", default="/tmp/granite_patchtst",
                    help="Path to local clone of the granite-tsfm patchtst-fm branch.")
    ap.add_argument("--gift-eval-storage", default=None,
                    help="Path to GIFT-EVAL data root. Overrides $GIFT_EVAL. "
                         "If unset, falls back to the $GIFT_EVAL env var.")
    ap.add_argument("--out-csv", default=str(SUBMIT_DIR / "all_results.csv"))
    ap.add_argument("--no-domain-info", action="store_true",
                    help="Skip dataset_properties.json lookup (leave domain/num_variates blank).")
    ap.add_argument("--no-multivariate", action="store_true",
                    help="Force to_univariate=True everywhere (split multivar into univariate series before inference).")
    ap.add_argument("--terms", nargs="+", default=["short", "medium", "long"],
                    choices=["short", "medium", "long"],
                    help="Which forecast terms to evaluate. Default: all three.")
    ap.add_argument("--sundial-python",
                    default=os.environ.get("SUNDIAL_PYTHON", "python"),
                    help="Path to the sundial env's python (Sundial subprocess). "
                         "Defaults to $SUNDIAL_PYTHON, else 'python' on PATH.")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="Split (ds, term) configs into N roughly-equal shards, "
                         "process only --shard K (1-indexed). Used for multi-GPU parallelism: "
                         "launch the same command 4x with CUDA_VISIBLE_DEVICES=0..3 and "
                         "--num-shards 4 --shard 1..4.")
    ap.add_argument("--shard", type=int, default=1,
                    help="Which shard to process (1..num_shards). Default: 1.")
    ap.add_argument("--tau-m", type=float, default=None,
                    help="Margin gate threshold. Defaults to predictor's TAU_M (0.07 for XGB).")
    ap.add_argument("--tau-d", type=float, default=None,
                    help="Diversity gate threshold. Defaults to predictor's TAU_D (0.07 for XGB).")
    args = ap.parse_args()
    if not (1 <= args.shard <= args.num_shards):
        ap.error(f"--shard must be in [1, {args.num_shards}], got {args.shard}")

    if args.gift_eval_storage:
        os.environ["GIFT_EVAL"] = args.gift_eval_storage
    if not os.environ.get("GIFT_EVAL"):
        ap.error("GIFT-EVAL data path not set. Pass --gift-eval-storage /path/to/GIFT-EVAL "
                 "or export GIFT_EVAL=/path/to/GIFT-EVAL.")
    log.info("GIFT_EVAL storage = %s", os.environ["GIFT_EVAL"])

    datasets = args.datasets or ALL_DATASETS
    log.info("Evaluating TimeRouter on %d datasets", len(datasets))

    dataset_props = {} if args.no_domain_info else _load_dataset_props()

    predictor_kwargs = dict(
        ckpt_dir=args.ckpt_dir,
        ckpt_repo=args.ckpt_repo,
        granite_patchtst_path=args.granite_patchtst,
        sundial_python=args.sundial_python,
    )
    if args.tau_m is not None:
        predictor_kwargs["tau_m"] = args.tau_m
    if args.tau_d is not None:
        predictor_kwargs["tau_d"] = args.tau_d

    # Enumerate all (ds_name, term) configs first, then shard.
    all_configs: List[tuple] = []
    for ds_name in datasets:
        for term in args.terms:
            if (term in ("medium", "long")) and ds_name not in MED_LONG_DATASETS:
                continue
            all_configs.append((ds_name, term))

    # Even split into args.num_shards. Round-robin assignment so each shard
    # gets a mix of small/large tasks (alternative would be contiguous chunks,
    # but task ordering is alphabetical-ish so contiguous would imbalance).
    my_configs = [
        (ds, term) for k, (ds, term) in enumerate(all_configs)
        if (k % args.num_shards) == (args.shard - 1)
    ]
    log.info("Sharding: shard %d/%d → %d / %d total configs",
             args.shard, args.num_shards, len(my_configs), len(all_configs))

    # Auto-tag the output CSV with shard info if multi-shard.
    out_csv = args.out_csv
    if args.num_shards > 1:
        from os.path import splitext
        stem, ext = splitext(out_csv)
        out_csv = f"{stem}_shard{args.shard}of{args.num_shards}{ext}"
        log.info("Output CSV (shard-tagged): %s", out_csv)

    rows: List[dict] = []
    for cfg_num, (ds_name, term) in enumerate(my_configs):
        log.info("[%d/%d  shard %d/%d] %s / %s",
                 cfg_num + 1, len(my_configs),
                 args.shard, args.num_shards, ds_name, term)
        t0 = time.time()
        try:
            metrics = evaluate_on_dataset(predictor_kwargs, ds_name, term,
                                          use_multivariate_data=not args.no_multivariate)
        except Exception as e:
            log.exception("FAILED on %s/%s: %s", ds_name, term, e)
            continue
        ds_config = _ds_config_key(ds_name, term)
        ds_key = ds_name.split("/")[0]
        ds_key = PRETTY_NAMES.get(ds_key, ds_key)
        props = dataset_props.get(ds_key, {})
        row = {
            "dataset": ds_config,
            "model": MODEL_NAME,
            **{f"eval_metrics/{k}": v for k, v in metrics.items()},
            "domain": props.get("domain", ""),
            "num_variates": props.get("num_variates", 1),
        }
        rows.append(row)
        log.info("    done in %.1fs   MASE=%.4f   wQL=%.4f",
                 time.time() - t0,
                 row.get("eval_metrics/MASE[0.5]", float("nan")),
                 row.get("eval_metrics/mean_weighted_sum_quantile_loss", float("nan")))

    expected_cols = [
        "dataset", "model",
        "eval_metrics/MSE[mean]", "eval_metrics/MSE[0.5]", "eval_metrics/MAE[0.5]",
        "eval_metrics/MASE[0.5]", "eval_metrics/MAPE[0.5]", "eval_metrics/sMAPE[0.5]",
        "eval_metrics/MSIS", "eval_metrics/RMSE[mean]", "eval_metrics/NRMSE[mean]",
        "eval_metrics/ND[0.5]", "eval_metrics/mean_weighted_sum_quantile_loss",
        "domain", "num_variates",
    ]
    df = pd.DataFrame(rows)
    for c in expected_cols:
        if c not in df.columns:
            df[c] = ""
    df = df[expected_cols].sort_values(by="dataset")
    df.to_csv(out_csv, index=False)
    log.info("Wrote %s (%d rows + header, %d cols)", out_csv, len(df), df.shape[1])

    # Compute LB MASE only when this run covers all 97 configs.
    # Per-shard LB is meaningless (different denominators); merge shards first
    # via gift_eval/merge_shards.py, then compute LB on the merged CSV.
    if args.num_shards == 1:
        try:
            sn = pd.read_csv(SN_CSV).drop_duplicates(subset="dataset", keep="last")
            sn_map = sn.set_index("dataset")["eval_metrics/MASE[0.5]"]
            paired = (df.set_index("dataset")["eval_metrics/MASE[0.5]"].astype(float)
                        / sn_map).replace([np.inf, -np.inf], np.nan).dropna()
            lb = float(gmean(paired.values))
            (SUBMIT_DIR / "lb_mase.txt").write_text(f"{lb:.4f}\n")
            log.info("LB MASE = %.4f  (deployed = 0.6746)", lb)
        except Exception as e:
            log.warning("Could not compute LB MASE: %s", e)
    else:
        log.info("Per-shard LB MASE intentionally skipped. After all shards finish, run:")
        log.info("    python gift_eval/merge_shards.py 'gift_eval/all_results_shard*of%d.csv' gift_eval/all_results.csv",
                 args.num_shards)


if __name__ == "__main__":
    main()
