"""TimeRouterPredictor — inline routing predictor for GIFT-EVAL.

Mirrors the official chronos-2 notebook predictor interface
(https://github.com/SalesforceAIResearch/gift-eval/blob/main/notebooks/chronos-2.ipynb):
takes a list of test_data.input items and returns a list of gluonts
QuantileForecast objects.

For multivariate datasets (target shape (V, T) when ``use_multivariate_data=True``):

  - Chronos-2 runs in its native multivariate mode -- one batched call over
    items, each with V variates jointly inferred (Chronos-2 was pretrained
    with cross-variate attention).
  - PatchTST-FM and Sundial are univariate-only foundation models; we split
    each (V, T) item into V univariate series, run them through the existing
    chunked univariate path, then re-stack the V outputs.
  - Routing is done at the **per-variate** level so the 270-dim feature
    distribution stays aligned with the router's training data (which was
    generated with ``to_univariate=True`` per ``run_fm_inference.py``).
  - Output is reassembled into per-item multivariate QuantileForecast with
    shape (n_keys, V, h).

Per item, the steps are:
  1. Run all 3 FMs on the input context  → 9-quantile future-horizon forecast.
  2. Run all 3 FMs on context[:-h]        → CV backtest.
  3. cv_score = MASE(cv_pred, context[-h:], scale)  per FM.
  4. Build the deployed 270-dim feature vector (ts_stats + cv block + ctx +
     regime_shift + raw_series + 32-step forecast preview per FM).
  5. Apply the shipped 5-seed OvA ensemble → margin = top1 − top2 softmax.
  6. Diversity = mean over horizon of std across z-normalized FM forecasts.
  7. Combined-OR gate at (tau_m, tau_d) = (0.07, 0.07): defer to
     CV-inverse-weighted ensemble; otherwise argmax FM pick.
  8. Output: QuantileForecast with the chosen 9-quantile array (univariate)
     or stacked variate decisions (multivariate).

The 3 FM wrappers are reused from ``final_scripts.pipeline.run_fm_inference``;
no precomputed features or forecasts are read from disk.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gluonts.model.forecast import QuantileForecast  # noqa: E402

# Chronos-2 + PatchTST-FM run inline (transformers 4.57+ env). Sundial is
# subprocess'd into the sundial env (transformers 4.40.1) -- see
# gift_eval/_sundial_worker.py for the why.
from gift_eval._fm.fm_predictors import (  # noqa: E402
    PatchTSTFMPredictor, RegistryFMPredictor, FlowStatePredictor,
    _dataset_meta as _flowstate_dataset_meta,
)

from gift_eval._features import (  # noqa: E402
    POOL, PRED_LEN, build_feature_vector, compute_diversity, feature_order,
)
from gift_eval._router import (  # noqa: E402
    DEFAULT_CKPT_REPO, SEEDS, TAU_D, TAU_M,
    cv_inverse_weighted_quantiles, decide, load_ensemble,
    predict_proba_ova, resolve_ckpt_dir,
)

QUANTILE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
SUBMIT_CKPT_DIR = Path(__file__).resolve().parent / "checkpoints_k4_chronos_flowstate_patchtst_fm_sundial"
SUBMIT_DIR = Path(__file__).resolve().parent

# Default path to the sundial env's Python interpreter for the Sundial subprocess.
# Override per-call via TimeRouterPredictor(..., sundial_python=...) or $SUNDIAL_PYTHON.
DEFAULT_SUNDIAL_PYTHON = os.environ.get("SUNDIAL_PYTHON", "python")

logger = logging.getLogger(__name__)


def _per_row_mase(forecast: np.ndarray, target: np.ndarray, scale: float) -> float:
    """MASE for one window: mean |target − forecast| / scale, NaN-safe."""
    if scale is None or not np.isfinite(scale) or scale <= 0:
        return float("nan")
    f = np.asarray(forecast, dtype=np.float64)
    t = np.asarray(target,   dtype=np.float64)
    if f.shape != t.shape or np.any(~np.isfinite(f)):
        return float("nan")
    diff = np.abs(t - f)
    valid = np.isfinite(diff)
    if not valid.any():
        return float("nan")
    return float(diff[valid].mean()) / scale


def _seasonal_error(target: np.ndarray, seasonality: int) -> float:
    """gluonts.ev.ts_stats.seasonal_error, NaN-safe (mask invalid lag-diffs)."""
    target = np.asarray(target, dtype=np.float64)
    if target.ndim > 1:
        target = target.flatten()
    if len(target) <= seasonality:
        if len(target) <= 1:
            return float("nan")
        d = np.abs(np.diff(target))
        d = d[np.isfinite(d)]
        return float(d.mean()) if len(d) else float("nan")
    diff = np.abs(target[seasonality:] - target[:-seasonality])
    diff = diff[np.isfinite(diff)]
    return float(diff.mean()) if len(diff) else float("nan")


class TimeRouterPredictor:
    """gluonts-compatible predictor wrapping the deployed K=3 router.

    Lazy-init: FM wrappers are constructed on first ``predict`` call so that
    swap-in/out across notebook cells is cheap and a fresh process can pick
    the right CUDA device via env vars.
    """

    # FM defaults align with final_scripts/pipeline/run_fm_inference.py
    # (which itself mirrors the official GIFT-EVAL notebooks). Centralised here
    # so the user can see/audit the deployed inference settings at a glance.
    DEFAULT_FM_KWARGS = {
        "chronos": {
            "repo_id": "amazon/chronos-2",
            "batch_size": 1024,                   # BATCH_SIZE_DEFAULTS["chronos"]
            # predict_batches_jointly=True is forced by RegistryFMPredictor.EXTRA_KWARGS
            # — leakage-safe under chronos-2.ipynb iteration (one window per call).
        },
        "patchtst_fm": {
            "model_path": "ibm-research/patchtst-fm-r1",
            "batch_size": 2048,                   # official notebook
        },
        "sundial": {
            "model_path": "thuml/sundial-base-128m",
            "batch_size": 1024,                   # official notebook
            "num_samples": 100,                   # quantiles derived from 100 samples
            "batch_x_shape": 2880,                # official notebook clip length
        },
        "flowstate": {
            "model_path": "ibm-research/FlowState",
            "batch_size": 16,                     # official notebook
        },
    }

    def __init__(
        self,
        prediction_length: int,
        freq: str,
        seasonality: int,
        ckpt_dir: str | Path = SUBMIT_CKPT_DIR,
        ckpt_repo: str = DEFAULT_CKPT_REPO,
        tau_m: float | None = None,
        tau_d: float | None = None,
        granite_patchtst_path: str = "/tmp/granite_patchtst",
        device: str = "auto",
        fm_kwargs: dict | None = None,
        sundial_python: str = DEFAULT_SUNDIAL_PYTHON,
        ds_name: str = "",
    ):
        self.prediction_length = int(prediction_length)
        self.freq = freq
        self.seasonality = int(seasonality)
        # Resolve to a local dir: use ckpt_dir if it holds weights, else download from
        # HuggingFace (ckpt_dir as a repo id, or ckpt_repo as the missing-local fallback).
        self.ckpt_dir = resolve_ckpt_dir(ckpt_dir, ckpt_repo)
        # Resolve tau from pool_metadata.json if not explicitly passed.
        meta_path = self.ckpt_dir / "pool_metadata.json"
        if (tau_m is None or tau_d is None) and meta_path.exists():
            import json as _json
            meta = _json.load(open(meta_path))
            if tau_m is None: tau_m = float(meta.get("tau_m", TAU_M))
            if tau_d is None: tau_d = float(meta.get("tau_d", TAU_D))
        self.tau_m = float(TAU_M if tau_m is None else tau_m)
        self.tau_d = float(TAU_D if tau_d is None else tau_d)
        self._granite_path = granite_patchtst_path
        self._device = device
        self._sundial_python = sundial_python
        self.ds_name = ds_name
        # Resolve FM kwargs: defaults overridable per-FM via the fm_kwargs dict.
        self._fm_kwargs = {fm: dict(self.DEFAULT_FM_KWARGS[fm]) for fm in POOL}
        if fm_kwargs:
            for fm, ovrd in fm_kwargs.items():
                if fm in self._fm_kwargs and ovrd:
                    self._fm_kwargs[fm].update(ovrd)

        # FlowState requires (domain, no_daily) per dataset; resolve once.
        if "flowstate" in POOL:
            self._flow_domain, self._flow_no_daily = _flowstate_dataset_meta(ds_name)
        else:
            self._flow_domain, self._flow_no_daily = None, False

        # Lazy: instantiated on first predict() call.
        self._chronos = None
        self._patchtst = None
        self._flowstate = None
        self._ensemble = None

    def update_for_dataset(self, ds_name: str, freq: str, prediction_length: int,
                            seasonality: int):
        """Re-target this predictor at a new (ds_name, freq, h, season) without
        reloading FMs. Used by run_eval.py to avoid the tsfm_public sys.path
        conflict that arises if FlowState + PatchTSTFM init are interleaved
        across short-lived predictor instances."""
        self.ds_name = ds_name
        self.freq = freq
        self.prediction_length = int(prediction_length)
        self.seasonality = int(seasonality)
        if "flowstate" in POOL:
            self._flow_domain, self._flow_no_daily = _flowstate_dataset_meta(ds_name)

    # ------------------------------------------------------------------
    #  Lazy init
    # ------------------------------------------------------------------
    def _ensure_models(self):
        if self._chronos is None:
            kw = self._fm_kwargs["chronos"]
            logger.info("Loading Chronos-2 (%s, batch_size=%s) ...",
                        kw["repo_id"], kw.get("batch_size"))
            self._chronos = RegistryFMPredictor("chronos", **kw)
        # CRITICAL: load FlowState BEFORE PatchTST. Both use a `tsfm_public`
        # package, but from different git clones with subtly different model
        # internals (verified: same input → max abs diff ≈ 3.4 in point
        # forecast). The clone path that v4_fixed inference (Phase D source
        # forecasts) used is `/tmp/granite_tsfm_clone` (FlowState branch).
        # We must mirror that here, otherwise submit's LB MASE diverges from
        # the cached Phase D replay numbers. We load FlowState first (caches
        # older tsfm_public), then clear sys.modules so PatchTST loads its
        # own newer tsfm_public from /tmp/granite_patchtst. The already-
        # instantiated FlowState model object holds its class refs and is
        # unaffected by the cache clear.
        if "flowstate" in POOL and self._flowstate is None:
            kw = self._fm_kwargs["flowstate"]
            logger.info("Loading FlowState (%s, batch_size=%s) ...",
                        kw["model_path"], kw.get("batch_size"))
            self._flowstate = FlowStatePredictor(**kw, device=self._device)
            # Drop tsfm_public so PatchTST gets a fresh import from its own clone.
            import sys as _sys
            for _mod in list(_sys.modules):
                if _mod == "tsfm_public" or _mod.startswith("tsfm_public."):
                    del _sys.modules[_mod]
        if self._patchtst is None:
            kw = self._fm_kwargs["patchtst_fm"]
            logger.info("Loading PatchTST-FM (%s, batch_size=%s) ...",
                        kw["model_path"], kw.get("batch_size"))
            self._patchtst = PatchTSTFMPredictor(
                **kw, patchtst_repo_path=self._granite_path, device=self._device,
            )
        # Sundial is NOT loaded in-process. It is dispatched via subprocess to
        # the sundial env (transformers==4.40.1) -- see _run_sundial_subprocess.
        if self._ensemble is None:
            logger.info("Loading 5-seed OvA ensemble from %s ...", self.ckpt_dir)
            self._ensemble = load_ensemble(self.ckpt_dir)

    # ------------------------------------------------------------------
    #  Sundial subprocess bridge
    # ------------------------------------------------------------------
    def _run_sundial_subprocess(self, contexts: List[np.ndarray],
                                 cv_contexts: List[np.ndarray], horizon: int):
        """Run Sundial future + CV in a subprocess (sundial env).

        Returns (future_points, future_quantiles, cv_points). Each item is a
        list of np.ndarray, indexed by row.
        """
        import pickle
        import subprocess
        import tempfile
        kw = self._fm_kwargs["sundial"]
        with tempfile.TemporaryDirectory() as tmp:
            in_path  = Path(tmp) / "sundial_in.pkl"
            out_path = Path(tmp) / "sundial_out.pkl"
            payload = {
                "contexts":     contexts,
                "cv_contexts":  cv_contexts,
                "horizon":      horizon,
                "freq":         self.freq,
                "batch_size":   kw.get("batch_size", 1024),
                "num_samples":  kw.get("num_samples", 100),
                "batch_x_shape": kw.get("batch_x_shape", 2880),
            }
            pickle.dump(payload, open(in_path, "wb"))
            cmd = [
                self._sundial_python,
                str(SUBMIT_DIR / "_sundial_worker.py"),
                "--input",  str(in_path),
                "--output", str(out_path),
            ]
            logger.info("Dispatching Sundial subprocess (%d future + %d cv contexts) ...",
                        len(contexts), len(cv_contexts))
            try:
                # 30 min hard timeout per task -- the largest GIFT-EVAL config
                # (LOOP_SEATTLE/H short = 19 windows × 323 series) finishes in
                # ~6 min, so 30 min gives a 5x safety margin against a hang.
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            except subprocess.TimeoutExpired:
                raise RuntimeError(
                    "Sundial subprocess timeout (>30 min) -- worker likely hung. "
                    f"Input pickle: {in_path}, n_contexts={len(contexts)}.")
            if r.returncode != 0:
                logger.error("Sundial subprocess STDERR:\n%s", r.stderr[-2000:])
                raise RuntimeError(f"Sundial subprocess failed (exit {r.returncode})")
            res = pickle.load(open(out_path, "rb"))
        return res["future_points"], res["future_quantiles"], res["cv_points"]

    # ------------------------------------------------------------------
    #  Batched FM inference  (matches final_scripts/pipeline/run_fm_inference)
    # ------------------------------------------------------------------
    INNER_CHUNK = 256   # match chunk_size in run_fm_inference.run_inference_split

    def _fm_handle(self, name: str):
        # Sundial is not in-process; callers must use _run_sundial_subprocess instead.
        return {"chronos": self._chronos,
                "patchtst_fm": self._patchtst,
                "flowstate": self._flowstate}[name]

    def _run_pool_aligned_subset(self, contexts: List[np.ndarray],
                                  cv_contexts: List[np.ndarray],
                                  horizon: int,
                                  fm_names: List[str]):
        """Per-FM serial chunked future+CV inference for in-process FMs.

        For each in-process FM: for each chunk of INNER_CHUNK rows: future_pass, cv_pass.
        Sundial is dispatched via ``_run_sundial_subprocess`` (separate env).
        """
        self._ensure_models()
        n = len(contexts)
        future = {fm: ([None] * n, [None] * n) for fm in fm_names}
        cv_back = {fm: [None] * n for fm in fm_names}

        for fm_name in fm_names:
            if fm_name == "sundial":
                f_pts, f_qs, cv_pts = self._run_sundial_subprocess(
                    contexts, cv_contexts, horizon)
                future[fm_name] = (list(f_pts), list(f_qs))
                cv_back[fm_name] = list(cv_pts)
                continue
            fm = self._fm_handle(fm_name)
            # FlowState needs (domain, no_daily) — without these the gift_wrapper
            # uses a wrong scale_factor (we identified this as the root cause of
            # +186 bp v3→broken-v4 regression). Resolved in __init__ from ds_name.
            fm_extra = {}
            if fm_name == "flowstate":
                fm_extra = {"domain": self._flow_domain,
                            "no_daily": self._flow_no_daily}
            for cstart in range(0, n, self.INNER_CHUNK):
                ce = min(cstart + self.INNER_CHUNK, n)
                idxs = list(range(cstart, ce))
                ctx_chunk = [contexts[i]    for i in idxs]
                cv_chunk  = [cv_contexts[i] for i in idxs]
                f_pts, f_qs = fm.predict_batch(
                    [c.copy() for c in ctx_chunk], horizon, freq=self.freq, **fm_extra)
                cv_pts, _   = fm.predict_batch(
                    [c.copy() for c in cv_chunk],  horizon, freq=self.freq, **fm_extra)
                for k, i in enumerate(idxs):
                    future[fm_name][0][i] = f_pts[k]
                    future[fm_name][1][i] = f_qs[k]
                    cv_back[fm_name][i]   = cv_pts[k]
        return future, cv_back

    # ------------------------------------------------------------------
    #  Multivariate Chronos-2 path
    #  Mirrors notebooks/chronos-2.ipynb Chronos2Predictor.predict():
    #    inputs = [{"target": ...}, ...]  ← list of dicts, not a padded tensor
    #    pipeline.predict_quantiles(inputs=..., predict_batches_jointly=True,
    #                                batch_size=...)
    #  Pipeline handles internal padding/batching/OOM-recovery.
    # ------------------------------------------------------------------
    def _ensure_chronos_native(self):
        """Lazy-load BaseChronosPipeline directly so we can pass multivar inputs.

        Pin device_map to "cuda:0" when CUDA is available -- mirrors the
        timecopilot wrapper (timecopilot/models/foundation/chronos.py:198).
        Without this, BaseChronosPipeline.from_pretrained defaults to CPU
        and inference is ~50x slower.
        """
        if getattr(self, "_chronos_native", None) is None:
            import torch as _torch
            from chronos import BaseChronosPipeline
            repo = self._fm_kwargs["chronos"]["repo_id"]
            device_map = "cuda:0" if _torch.cuda.is_available() else "cpu"
            logger.info("Loading Chronos-2 native pipeline (%s) on %s for multivar inference ...",
                        repo, device_map)
            self._chronos_native = BaseChronosPipeline.from_pretrained(
                repo, device_map=device_map)
            self._torch_chronos = _torch

    def _chronos_native_predict(self, multivar_ctxs: List[np.ndarray], horizon: int,
                                 quantile_levels=QUANTILE_LEVELS):
        """Match chronos-2.ipynb's predict() exactly.

        multivar_ctxs: list of n_items arrays. Each is 1D (T,) for univariate
                       items or 2D (V, T) for multivariate items.
        Returns:
            point_per_item:    list of n_items arrays, each (h,) or (V, h)
            quantile_per_item: list of n_items arrays, each shaped per the
                               notebook convention -- (n_quantiles, h) for
                               univariate or (n_quantiles, h, V) for multivariate.
        """
        self._ensure_chronos_native()
        torch = self._torch_chronos
        # Build the list-of-dicts input (notebook _pack_model_items).
        inputs = [{"target": np.ascontiguousarray(mc, dtype=np.float32)}
                  for mc in multivar_ctxs]
        is_univariate = inputs[0]["target"].ndim == 1
        bs = self._fm_kwargs["chronos"].get("batch_size", 1024)

        # OOM-fallback loop, matching the notebook.
        while True:
            try:
                quantiles_list, mean_list = self._chronos_native.predict_quantiles(
                    inputs=inputs,
                    prediction_length=horizon,
                    batch_size=bs,
                    quantile_levels=list(quantile_levels),
                    predict_batches_jointly=True,
                )
                break
            except torch.cuda.OutOfMemoryError:
                bs = max(1, bs // 2)
                logger.warning("Chronos-2 OOM at batch_size > %d, halving", bs)

        # Stack to one tensor: (batch, V, h, n_quantiles)
        q_stacked = torch.stack(quantiles_list)
        # Permute to (batch, n_quantiles, h, V) per notebook convention.
        q_stacked = q_stacked.permute(0, 3, 2, 1).cpu().numpy()
        if is_univariate:
            q_stacked = q_stacked.squeeze(-1)  # (batch, n_quantiles, h)

        # Mean: list of (V, h) or (h,) tensors.
        mean_arrays = [m.cpu().numpy() for m in mean_list]
        return mean_arrays, [q_stacked[i] for i in range(len(inputs))]

    # ------------------------------------------------------------------
    #  Predict
    # ------------------------------------------------------------------
    def predict(self, test_data_input) -> List[QuantileForecast]:
        items = list(test_data_input)
        if not items:
            return []
        h = self.prediction_length

        # Per-item meta (start_date, item_id, V) and per-variate flat lists.
        per_item_meta: List[tuple] = []
        multivar_ctxs: List[np.ndarray] = []          # each (V, T)
        flat_ctxs: List[np.ndarray] = []              # each (T,)
        flat_back: List[tuple] = []                   # (item_idx, var_idx)
        for it_i, it in enumerate(items):
            target = np.asarray(it["target"], dtype=np.float64)
            if target.ndim == 1:
                target = target[None, :]
            V, T = target.shape
            per_item_meta.append((it.get("start"), it.get("item_id", ""), V, T))
            multivar_ctxs.append(target)
            for v in range(V):
                flat_ctxs.append(target[v])
                flat_back.append((it_i, v))

        n_flat = len(flat_ctxs)
        n_items = len(items)
        any_multivar = any(m[2] > 1 for m in per_item_meta)
        # Precompute (item_idx, var_idx) -> flat_idx for O(1) lookup; the prior
        # `next(... for k in enumerate(flat_back) if ...)` was O(n_flat) per
        # query and quadratic over big multivar tasks (bitbrains 2500 rows).
        flat_lookup = {pair: k for k, pair in enumerate(flat_back)}
        item_to_flat_indices: List[List[int]] = [[] for _ in range(n_items)]
        for k, (it_i, _) in enumerate(flat_back):
            item_to_flat_indices[it_i].append(k)

        # CV per variate (for cv_score using existing univariate cv_targets).
        cv_ctxs_flat    = [c[:-h] if len(c) > h else c.copy() for c in flat_ctxs]
        cv_targets_flat = [c[-h:] if len(c) > h else c.copy() for c in flat_ctxs]

        # Storage indexed by flat row.
        future = {fm: ([None] * n_flat, [None] * n_flat) for fm in POOL}
        cv_back = {fm: [None] * n_flat for fm in POOL}

        self._ensure_models()

        # ---------------- Chronos: multivariate per item (notebook-aligned) ----
        if any_multivar:
            # Future-horizon pass over native multivar inputs.
            mean_per_item, q_per_item = self._chronos_native_predict(
                multivar_ctxs, h, quantile_levels=QUANTILE_LEVELS)
            # CV pass: drop last h timesteps from each item's (V, T) context.
            cv_multivar_ctxs = [mc[:, :-h] if mc.shape[1] > h else mc.copy()
                                for mc in multivar_ctxs]
            cv_mean_per_item, _ = self._chronos_native_predict(
                cv_multivar_ctxs, h, quantile_levels=[0.5])
            # Unpack to per-flat-row (per-(item, variate)) outputs.
            #   mean_per_item[i] shape: (V, h) for multivar items, (h,) for univar.
            #   q_per_item[i]    shape: (n_q, h, V) for multivar, (n_q, h) for univar.
            for it_i, (start, iid, V, _) in enumerate(per_item_meta):
                m_i = np.asarray(mean_per_item[it_i])
                q_i = np.asarray(q_per_item[it_i])
                cv_m_i = np.asarray(cv_mean_per_item[it_i])
                for v in range(V):
                    flat_idx = flat_lookup[(it_i, v)]
                    if V == 1 and m_i.ndim == 1:
                        future["chronos"][0][flat_idx] = m_i               # (h,)
                        future["chronos"][1][flat_idx] = q_i                # (n_q, h)
                        cv_back["chronos"][flat_idx]   = cv_m_i             # (h,)
                    else:
                        future["chronos"][0][flat_idx] = m_i[v]            # (h,)
                        # q_i shape (n_q, h, V) -> (n_q, h) for variate v.
                        future["chronos"][1][flat_idx] = q_i[..., v]
                        cv_back["chronos"][flat_idx]   = cv_m_i[v]         # (h,)
        else:
            # Pure univariate path: use the standard chunked helper.
            tmp_future, tmp_cv = self._run_pool_aligned_subset(
                flat_ctxs, cv_ctxs_flat, h, ["chronos"])
            future["chronos"] = tmp_future["chronos"]
            cv_back["chronos"] = tmp_cv["chronos"]

        # ---------------- PatchTST / Sundial / FlowState: per-variate ----------
        non_chronos = [fm for fm in POOL if fm != "chronos"]
        tmp_future, tmp_cv = self._run_pool_aligned_subset(
            flat_ctxs, cv_ctxs_flat, h, non_chronos)
        for fm in non_chronos:
            future[fm] = tmp_future[fm]
            cv_back[fm] = tmp_cv[fm]

        # ---------------- Per-flat-row routing (per-variate decisions) ---------
        scales = [_seasonal_error(c, self.seasonality) for c in flat_ctxs]
        cv_scores_per_row: List[Dict[str, float]] = []
        for i in range(n_flat):
            row: Dict[str, float] = {}
            for fm in POOL:
                cv_pts = cv_back[fm][i]
                row[fm] = _per_row_mase(cv_pts, cv_targets_flat[i], scales[i])
            cv_scores_per_row.append(row)

        # Match training-time context truncation: Phase D's pool_features
        # were built on `base.parquet[context]` which is the LAST `raw_ctx_len`
        # points of the full series (NaN-preserving truncation, same as
        # make_base_<N>.py). For K=4 raw4096 ckpt, raw_ctx_len = 4096.
        # build_feature_vector consumes this for ts_stats + raw_series + ctx
        # blocks; forecasts (point_fcsts) are kept as-is (generated from full
        # ctx upstream, matches Phase B's per_fm parquet generation).
        FEAT_CTX_LEN = 4096
        X = np.zeros((n_flat, len(feature_order(POOL))), dtype=np.float32)
        for i in range(n_flat):
            point_fcsts = {fm: future[fm][0][i] for fm in POOL}
            ctx_for_feat = flat_ctxs[i][-FEAT_CTX_LEN:] if len(flat_ctxs[i]) > FEAT_CTX_LEN else flat_ctxs[i]
            X[i] = build_feature_vector(
                ctx_for_feat, h, self.freq, point_fcsts, cv_scores_per_row[i])

        proba = predict_proba_ova(self._ensemble, X, K=len(POOL))
        diversity = np.zeros(n_flat, dtype=np.float64)
        for i in range(n_flat):
            point_fcsts = {fm: future[fm][0][i] for fm in POOL}
            ctx_for_div = flat_ctxs[i][-FEAT_CTX_LEN:] if len(flat_ctxs[i]) > FEAT_CTX_LEN else flat_ctxs[i]
            diversity[i] = compute_diversity(ctx_for_div, point_fcsts)

        d = decide(proba, diversity, tau_m=self.tau_m, tau_d=self.tau_d)
        picks, gated = d["picks"], d["gated"]

        # Per-flat-row final quantile arrays (each (n_quantiles, h)).
        # NaN guard: if the router picks a FM whose forecast contains NaN
        # (e.g. Sundial's RuntimeError fallback), fall back to the
        # CV-inverse-weighted ensemble over the valid FMs. If the ensemble itself
        # would be NaN (all FMs broken on this row -- which is rare), we use
        # the cleanest single FM available.
        final_q_flat: List[np.ndarray] = []
        for i in range(n_flat):
            quantile_arrays = {fm: future[fm][1][i] for fm in POOL}
            valid_qs = {
                fm: q for fm, q in quantile_arrays.items()
                if q is not None and np.all(np.isfinite(np.asarray(q)))
            }
            if gated[i]:
                final_q = cv_inverse_weighted_quantiles(valid_qs, cv_scores_per_row[i])
            else:
                fm_pick = POOL[int(picks[i])]
                if fm_pick in valid_qs:
                    final_q = valid_qs[fm_pick]
                elif valid_qs:
                    # Picked FM is NaN; defer to ensemble of valid FMs.
                    final_q = cv_inverse_weighted_quantiles(valid_qs, cv_scores_per_row[i])
                else:
                    # Worst case: every FM NaN on this row. Fall back to a context-level
                    # naive (last value broadcast) so evaluate_forecasts doesn't choke.
                    last_val = float(np.nan_to_num(np.nanmean(flat_ctxs[i]), nan=0.0))
                    final_q = np.full((len(QUANTILE_LEVELS), h), last_val, dtype=np.float32)
            final_q_flat.append(np.asarray(final_q, dtype=np.float32))

        # ---------------- Re-assemble per-item Forecasts ----------------------
        # Notebook convention: forecast_arrays shape is
        #   univariate:    (n_quantiles, h)
        #   multivariate:  (n_quantiles, h, V)
        # forecast_keys = list(map(str, quantile_levels))   -- no "mean" key.
        #
        # We use gluonts.start_date semantics matching the notebook:
        #   forecast_start_date = ts["start"] + len(ts["target"])
        forecasts: List[QuantileForecast] = []
        forecast_keys = [str(q) for q in QUANTILE_LEVELS]
        for it_i, (start, iid, V, T) in enumerate(per_item_meta):
            # item_to_flat_indices already iterated in (item, variate) insertion order.
            flat_indices = item_to_flat_indices[it_i]
            stacked_q = [final_q_flat[k] for k in flat_indices]
            if V == 1:
                forecast_arrays = stacked_q[0]                            # (n_q, h)
            else:
                # (V, n_q, h) -> (n_q, h, V)
                forecast_arrays = np.transpose(np.stack(stacked_q, axis=0), (1, 2, 0))
            forecast_arrays = np.asarray(forecast_arrays, dtype=np.float32)

            # Forecast start = end-of-context for this item.
            try:
                fc_start = start + T   # gluonts Period + int -> Period
            except Exception:
                fc_start = start

            forecasts.append(QuantileForecast(
                forecast_arrays=forecast_arrays,
                forecast_keys=forecast_keys,
                start_date=fc_start,
                item_id=iid,
            ))
        return forecasts
