#!/usr/bin/env python3
"""Sundial inference worker.

Runs in the ``sundial`` env (transformers==4.40.1), invoked as a subprocess
from the main predictor (which lives in the ``timerouter`` env, transformers==4.57).

Sundial's HuggingFace ``trust_remote_code`` module imports
``validate_stopping_criteria`` from ``transformers.generation`` -- this symbol
was removed in transformers 4.50, so Sundial only loads under transformers 4.40.x.
PatchTST-FM in turn produces all-NaN forecasts under transformers 4.40.x but
works under transformers 4.57. The two FMs cannot coexist in one Python env
(see the README). This worker keeps Sundial isolated.

Protocol (pickle in / pickle out):
    Input  : {"contexts": [np.ndarray, ...], "cv_contexts": [...],
              "horizon": int, "freq": str,
              "batch_size": int, "num_samples": int, "batch_x_shape": int}
    Output : {"future_points": [(h,) ndarray, ...],
              "future_quantiles": [(9, h) ndarray, ...],
              "cv_points": [(h,) ndarray, ...]}

CLI:
    python gift_eval/_sundial_worker.py --input /tmp/sundial_in.pkl --output /tmp/sundial_out.pkl
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("sundial_worker")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    payload = pickle.load(open(args.input, "rb"))
    contexts    = payload["contexts"]
    cv_contexts = payload["cv_contexts"]
    horizon     = int(payload["horizon"])
    freq        = payload["freq"]
    bs          = int(payload.get("batch_size", 1024))
    n_samples   = int(payload.get("num_samples", 100))
    batch_x_shape = int(payload.get("batch_x_shape", 2880))

    from gift_eval._fm.fm_predictors import SundialPredictor
    log.info("Loading Sundial in subprocess (env transformers must be ~4.40.1) ...")
    sd = SundialPredictor(
        model_path="thuml/sundial-base-128m",
        batch_size=bs,
        num_samples=n_samples,
        batch_x_shape=batch_x_shape,
    )
    log.info("Sundial ready. Running future_pass on %d contexts ...", len(contexts))
    future_points, future_quantiles = sd.predict_batch(
        [c.copy() for c in contexts], horizon, freq=freq)
    log.info("Running cv_pass on %d contexts ...", len(cv_contexts))
    cv_points, _ = sd.predict_batch(
        [c.copy() for c in cv_contexts], horizon, freq=freq)

    out = {
        "future_points":    future_points,
        "future_quantiles": future_quantiles,
        "cv_points":        cv_points,
    }
    pickle.dump(out, open(args.output, "wb"))
    log.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
