#!/usr/bin/env python3
"""Vendored FM predictor classes for the self-contained gift_eval/ package.

Trimmed copy of ``final_scripts/pipeline/run_fm_inference.py`` keeping only the
four FMs in the deployed K=4 pool plus the FlowState dataset-meta helper:

    SundialPredictor     — thuml/sundial-base-128m  (run via subprocess)
    FlowStatePredictor   — ibm-research/FlowState   (granite-tsfm gift-flowstate)
    PatchTSTFMPredictor  — ibm-research/patchtst-fm-r1 (granite-tsfm patchtst-fm)
    RegistryFMPredictor  — chronos-2 via the vendored chronos backend
    _dataset_meta(ds)    — (domain, no_daily) lookup used by FlowState

The offline inference driver (run_inference_split / main) and the unused
Moirai2 / TabPFN-TS predictors are intentionally dropped — gift_eval/ only does
online inference through ``gift_eval/predictor.py``.
"""
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure the active python's bin directory is on PATH so torch.utils.cpp_extension
# can find `ninja` (needed by xLSTM kernels in TiRex). When invoked via abs path
# (``/<env>/bin/python ...``) the conda activate hook is bypassed and PATH may
# not contain the env, so we patch it here.
_env_bin = os.path.dirname(sys.executable)
if _env_bin and _env_bin not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = _env_bin + os.pathsep + os.environ.get("PATH", "")

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


QUANTILE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
QUANTILE_COLS = lambda fm: [f"{fm}_quantile_{q:.1f}" for q in QUANTILE_LEVELS]


# =====================================================================
#  FM-specific predictors (each implements .predict_batch(contexts, horizon))
# =====================================================================

class _BasePredictor:
    name: str
    quantile_levels = QUANTILE_LEVELS

    def predict_batch(self, contexts: List[np.ndarray], horizon: int):
        """Return (point_forecasts, quantile_forecasts) for the batch.
        - point_forecasts: list of np.ndarray shape (horizon,)
        - quantile_forecasts: list of np.ndarray shape (9, horizon)
        """
        raise NotImplementedError


class SundialPredictor(_BasePredictor):
    """thuml/sundial-base-128m via transformers AutoModelForCausalLM.

    The model generates ``num_samples`` future trajectories per input. We
    compute quantiles across the sample dim to get a (9, horizon) array.
    """
    name = "sundial"

    def __init__(self, model_path="thuml/sundial-base-128m",
                 num_samples=100, batch_size=1024, batch_x_shape=2880,
                 device="auto"):
        # Defaults match the official notebook: num_samples=100, batch_size=1024,
        # batch_x_shape=2880 (clip context to last 2880 steps).
        import torch
        from transformers import AutoModelForCausalLM, set_seed
        from gluonts.transform import LastValueImputation
        set_seed(1)
        self._torch = torch
        self._LastValueImputation = LastValueImputation
        self.model_path = model_path
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.batch_x_shape = batch_x_shape
        self.device = (torch.device("cuda") if device == "auto" and torch.cuda.is_available()
                       else torch.device(device if device != "auto" else "cpu"))
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True
        ).to(self.device)
        self.model.eval()

    def _left_pad_and_stack_1D(self, tensors):
        torch = self._torch
        max_len = max(len(c) for c in tensors)
        padded = []
        for c in tensors:
            padding = torch.full(size=(max_len - len(c),), fill_value=torch.nan,
                                  device=c.device)
            padded.append(torch.concat((padding, c), dim=-1))
        return torch.stack(padded)

    def _prepare_context(self, contexts):
        torch = self._torch
        ts = [torch.tensor(c, dtype=torch.float32) for c in contexts]
        batch_x = self._left_pad_and_stack_1D(ts)
        if batch_x.shape[-1] > self.batch_x_shape:
            batch_x = batch_x[..., -self.batch_x_shape:]
        # NOTE: do NOT zero-pad to batch_x_shape (2880). The earlier zero-pad
        # was a workaround for a shape-error that only occurred under the
        # OLD env (transformers 4.57). With the dedicated `sundial`
        # env (transformers==4.40.1), Sundial accepts arbitrary input
        # lengths via its own internal patching. Zero-padding short series
        # (e.g., m4_yearly with ctx=30 → 2850 zeros + 30 reals = 99% noise)
        # caused +200% MASE inflation on m4_yearly/short and +43% on
        # electricity/W/short. Match the official notebook: just left-pad-NaN
        # to batch max + LastValueImputation. (FIX.7, May 4)
        if torch.isnan(batch_x).any():
            arr = batch_x.cpu().numpy()
            imputed = []
            for i in range(arr.shape[0]):
                imp = self._LastValueImputation()(arr[i])
                imputed.append(imp)
            arr = np.vstack(imputed)
            batch_x = torch.tensor(arr, dtype=torch.float32)
        return batch_x.contiguous().to(self.device)

    def predict_batch(self, contexts, horizon, freq=None):
        torch = self._torch
        out_samples = []
        bs = self.batch_size
        i = 0
        while i < len(contexts):
            j = min(i + bs, len(contexts))
            try:
                batch_x = self._prepare_context(contexts[i:j])
                with torch.autocast(device_type=self.device.type,
                                    dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32):
                    outputs = self.model.generate(
                        batch_x, max_new_tokens=horizon, revin=True,
                        num_samples=self.num_samples,
                    )
                out_samples.append(outputs.detach().float().cpu().numpy())
                i = j
            except torch.cuda.OutOfMemoryError:
                bs = max(1, bs // 2)
                torch.cuda.empty_cache()
                logger.warning("  Sundial OOM, reduced batch_size to %d", bs)
            except RuntimeError as e:
                # Catch any non-OOM RuntimeError (Sundial has several internal
                # shape/stride/position_ids issues for short or non-standard
                # input lengths). Fill NaN for these rows and advance.
                # OOM is already caught above as torch.cuda.OutOfMemoryError.
                n_skip = j - i
                nan_chunk = np.full((n_skip, self.num_samples, horizon),
                                     np.nan, dtype=np.float32)
                out_samples.append(nan_chunk)
                logger.warning("  Sundial RuntimeError on batch [%d:%d] (%d rows, ctx_len=%d) — filling NaN: %s",
                               i, j, n_skip,
                               batch_x.shape[-1] if 'batch_x' in dir() else -1,
                               str(e)[:160])
                i = j
        merged = np.concatenate(out_samples, axis=0)  # (N, num_samples, horizon)
        # Compute quantiles across sample axis
        q_levels = np.array(self.quantile_levels, dtype=np.float64)
        # merged shape: (N, num_samples, horizon)
        quantiles = np.quantile(merged, q_levels, axis=1)  # (9, N, horizon)
        quantiles = np.transpose(quantiles, (1, 0, 2))     # (N, 9, horizon)
        median_idx = self.quantile_levels.index(0.5)
        points = [quantiles[k, median_idx] for k in range(merged.shape[0])]
        quantile_list = [quantiles[k] for k in range(merged.shape[0])]
        return points, quantile_list


class FlowStatePredictor(_BasePredictor):
    """ibm-research/FlowState — quantile-output state-space TSFM.

    Uses the gift_wrapper logic from granite-tsfm@gift-flowstate so the
    behaviour matches the official LB submission.
    """
    name = "flowstate"

    def __init__(self, model_path="ibm-research/FlowState",
                 batch_size=16, device="auto",
                 granite_repo_path="/tmp/granite_tsfm_clone"):
        # Defaults match the official notebook: batch_size=16. FlowState's
        # gift_wrapper does dynamic batching internally based on context length,
        # so the static cap is just a safety upper bound.
        import torch
        sys.path.insert(0, granite_repo_path)
        from tsfm_public import FlowStateForPrediction
        from tsfm_public.models.flowstate.utils.utils import get_fixed_factor
        self._torch = torch
        self._get_fixed_factor = get_fixed_factor
        self.model_path = model_path
        self.batch_size = batch_size
        self.device = (torch.device("cuda") if device == "auto" and torch.cuda.is_available()
                       else torch.device(device if device != "auto" else "cpu"))
        self.model = FlowStateForPrediction.from_pretrained(model_path).to(self.device)
        self.model.eval()
        self.model.config.min_context = 0
        self.pretrain_context = self.model.config.context_length
        # quantiles in config order matches our QUANTILE_LEVELS [0.1, ..., 0.9]
        assert tuple(self.model.config.quantiles) == self.quantile_levels, \
            f"quantile mismatch: {self.model.config.quantiles}"
        # Frequency must be passed per call.
        self._freq = None
        self._domain = None

    def set_freq(self, freq: str, domain: str = None, no_daily: bool = False):
        self._freq = freq
        self._domain = domain
        self._no_daily = no_daily

    def _scale_factor(self):
        f = self._get_fixed_factor(self._freq, self._domain)
        if getattr(self, "_no_daily", False):
            # Match official gift-eval gift_wrapper.py:set_freq —
            # bizitobs_l2c has no daily cycle, only weekly.
            f /= 7
        return f

    @staticmethod
    def _fill_nan(seq: np.ndarray) -> np.ndarray:
        if not np.isnan(seq).any():
            return seq
        if not (~np.isnan(seq)).any():
            return np.zeros_like(seq)
        first_ix = np.isnan(seq).argmin()
        seq = seq[first_ix:]
        return seq  # match wrapper.replace_nan = False default

    def predict_batch(self, contexts, horizon, freq=None, domain=None, no_daily=False):
        torch = self._torch
        if freq is not None:
            self.set_freq(freq, domain=domain, no_daily=no_daily)
        if self._freq is None:
            raise RuntimeError("FlowStatePredictor: call set_freq(freq) or pass freq= to predict_batch")
        scale_factor = self._scale_factor()
        max_context = int(self.pretrain_context / scale_factor)
        # FlowState's SSM kernel computation needs L >= 2 in `kernel[:, -2]`.
        # The hard minimum is 2; previously this was set to 256 as an empirical
        # safety net, but that over-padded short series (e.g., electricity/W
        # ~132 effective tokens) with leading zeros, polluting the SSM warm-up
        # and causing +10.7% MASE inflation vs official on electricity/W/short.
        # MIN_LEN = 2 reproduces FlowState-9.1M baseline exactly (-0.001%).
        MIN_LEN = 2

        # Prepare per-row tensors with truncation + minimum-length padding.
        prepared = []
        for ctx in contexts:
            seq = self._fill_nan(np.asarray(ctx, dtype=np.float64))
            if len(seq) > max_context:
                seq = seq[-max_context:]
            if len(seq) < MIN_LEN:
                seq = np.concatenate([np.zeros(MIN_LEN - len(seq), dtype=seq.dtype), seq])
            prepared.append(seq)

        # Group by length, pad each group, run, then collect.
        # The original wrapper does dynamic batching by length cluster.
        order = sorted(range(len(prepared)), key=lambda i: len(prepared[i]))
        out_quantile = [None] * len(prepared)

        i = 0
        while i < len(order):
            cur_len = len(prepared[order[i]])
            # Dynamic batch size based on context length.
            bs = max(1, int(self.batch_size * self.pretrain_context / max(cur_len, 1)))
            j = i
            while j < len(order) and len(prepared[order[j]]) == cur_len and (j - i) < bs:
                j += 1
            group_idxs = order[i:j]
            seqs = [prepared[k] for k in group_idxs]
            # Stack as (seq_len, batch, 1)
            arr = np.stack(seqs, axis=0)             # (B, L)
            arr = arr.T[:, :, None]                  # (L, B, 1)
            x = torch.from_numpy(arr).float().to(self.device)
            with torch.no_grad():
                out = self.model(
                    past_values=x,
                    scale_factor=scale_factor,
                    prediction_length=horizon,
                    batch_first=False,
                )
            # New tsfm_public deprecates prediction_type='quantile' and forces
            # 'mean', leaving prediction_outputs as 3D mean and exposing
            # quantile_outputs as a separate 4D field. Older tsfm_public has
            # only prediction_outputs (4D quantile). Prefer quantile_outputs
            # when present.
            pred = getattr(out, "quantile_outputs", None)
            if pred is None:
                pred = out.prediction_outputs
            # pred shape: (B, num_quantiles, pred_len, num_channels)
            pred = pred.squeeze(-1).cpu().numpy()    # (B, 9, pred_len)
            # Optional positive-only constraint
            for kk, idx in enumerate(group_idxs):
                # Always allow nan-safe checks, never force positive here.
                out_quantile[idx] = pred[kk]
            i = j

        median_idx = self.quantile_levels.index(0.5)
        points = [q[median_idx] for q in out_quantile]
        return points, out_quantile


class PatchTSTFMPredictor(_BasePredictor):
    """ibm-research/patchtst-fm-r1 — quantile-output PatchTST-FM.

    Loaded from the patchtst-fm branch of granite-tsfm (must be cloned
    locally; default at /tmp/granite_patchtst).
    """
    name = "patchtst_fm"

    def __init__(self, model_path="ibm-research/patchtst-fm-r1",
                 batch_size=2048, device="auto",
                 patchtst_repo_path="/tmp/granite_patchtst"):
        # Defaults match the official notebook: batch_size=2048, model context
        # length 8192 (set internally by the model config).
        import torch, importlib
        # Force the cloned branch's tsfm_public to be used (it has patchtst_fm).
        if patchtst_repo_path not in sys.path:
            sys.path.insert(0, patchtst_repo_path)
        # Drop any previously-loaded tsfm_public so we get the branch version.
        for k in list(sys.modules):
            if k == "tsfm_public" or k.startswith("tsfm_public."):
                del sys.modules[k]
        from tsfm_public import PatchTSTFMForPrediction
        self._torch = torch
        self.batch_size = batch_size
        self.device = (torch.device("cuda") if device == "auto" and torch.cuda.is_available()
                       else torch.device(device if device != "auto" else "cpu"))
        self.model = PatchTSTFMForPrediction.from_pretrained(model_path,
                                                              device_map=str(self.device))
        self.model.eval()

    def predict_batch(self, contexts, horizon, freq=None):
        torch = self._torch
        out_quantiles = []
        bs = self.batch_size
        i = 0
        while i < len(contexts):
            j = min(i + bs, len(contexts))
            try:
                inputs = [torch.from_numpy(np.asarray(c, dtype=np.float32))
                          for c in contexts[i:j]]
                # Replace NaN with mean (preprocess from official wrapper).
                inputs = [
                    torch.from_numpy(np.nan_to_num(t.numpy(),
                                                    nan=float(np.nanmean(t.numpy()))
                                                    if (~np.isnan(t.numpy())).any() else 0.0))
                    if torch.isnan(t).any() else t
                    for t in inputs
                ]
                with torch.no_grad():
                    out = self.model(inputs=inputs,
                                     prediction_length=horizon,
                                     quantile_levels=list(self.quantile_levels))
                # quantile_predictions: list of (9, horizon) tensors
                for tens in out.quantile_predictions:
                    out_quantiles.append(tens.cpu().numpy())
                i = j
            except torch.cuda.OutOfMemoryError:
                bs = max(1, bs // 2)
                torch.cuda.empty_cache()
                logger.warning("  PatchTST-FM OOM, reduced batch_size to %d", bs)
        median_idx = self.quantile_levels.index(0.5)
        points = [q[median_idx] for q in out_quantiles]
        return points, out_quantiles


class RegistryFMPredictor(_BasePredictor):
    """Adapter for the FMs already registered in ``timeagents.models``
    (chronos / timesfm / tirex / ttm / auto_arima / etc.).

    Defaults (repo_id, batch_size) match the project's deployed configuration
    in ``configs/pool/*.yaml``. The wrappers' ``forecast(quantiles=...)``
    interface returns both the point and per-quantile columns in a single
    DataFrame; we adapt that to our (points, quantile) numpy interface.
    """
    name = "registry_fm"

    # repo_id defaults match _FOUNDATION_REPO_DEFAULTS in
    # timeagents.agents.forecaster (which mirrors configs/pool/3model.yaml).
    REPO_DEFAULTS = {
        "chronos":    "amazon/chronos-2",
        "timesfm":    "google/timesfm-2.5-200m-pytorch",
        "tirex":      "NX-AI/TiRex-1.1-gifteval",
        "ttm":        "ibm-granite/granite-timeseries-ttm-r2",
        "auto_arima": None,  # statistical, no repo
    }

    # Per-model kwargs to inject into get_model(). For auto_arima the registry
    # default (seasonal=True, nmodels=94) is pathologically slow on high-freq
    # data (5T/10T/15T) — full seasonal grid search can take hours per series.
    # The fast config matches what the labels_v3 baseline used: similar quality
    # at ~100x speed (Apr 23 finding).
    EXTRA_KWARGS = {
        "chronos": {
            # Restore official-equivalent `jointly=True` — safe because
            # run_inference_split() groups chronos rows by `window_idx`, so each
            # batch contains different SERIES at the SAME window position
            # (no cross-window leakage of the same series).
            "predict_batches_jointly": True,
        },
        "auto_arima": {
            "seasonal": False,
            "nmodels": 20,
            "max_p": 3,
            "max_q": 3,
            "approximation": True,
        },
    }

    def __init__(self, model_name: str, repo_id: str = None,
                 batch_size: int = None):
        from gift_eval._fm.chronos_backend import get_model, BATCH_SIZE_DEFAULTS
        kwargs = {}
        rid = repo_id or self.REPO_DEFAULTS.get(model_name)
        if rid is not None:
            kwargs["repo_id"] = rid
        bs = batch_size or BATCH_SIZE_DEFAULTS.get(model_name)
        if bs is not None:
            kwargs["batch_size"] = bs
        kwargs.update(self.EXTRA_KWARGS.get(model_name, {}))
        self.name = model_name
        self.model_name = model_name
        self.repo_id = rid
        self.batch_size = bs
        self.model = get_model(model_name, **kwargs)
        # Cache pandas + period helpers
        import pandas as pd
        self._pd = pd

    def predict_batch(self, contexts, horizon, freq=None):
        """Convert numpy contexts to (unique_id, ds, y) DataFrame, call the
        registry wrapper, then extract point + quantile arrays per row."""
        if freq is None:
            raise RuntimeError(f"{self.model_name}: predict_batch needs freq=")
        pd = self._pd
        # Build a single DataFrame for all contexts in this batch.
        # The timecopilot wrappers expect Timestamp (not Period) values for ds.
        unique_ids, datestamps, ys = [], [], []
        # Start in 1900 so yearly freq + long context (max ~266 years) doesn't
        # blow past pandas' nanosecond ceiling (~2262-04-11). The actual ds
        # values are only used for sorting / forecast tagging, not modeling.
        start = pd.Timestamp("1900-01-01")
        offset = pd.tseries.frequencies.to_offset(freq)
        for i, c in enumerate(contexts):
            arr = np.asarray(c, dtype=np.float64)
            n = len(arr)
            ids = np.full(n, i, dtype=np.int64)
            dates = pd.date_range(start=start, periods=n, freq=offset)
            unique_ids.append(ids)
            datestamps.append(dates)
            ys.append(arr)
        df = pd.DataFrame({
            "unique_id": np.concatenate(unique_ids),
            "ds": np.concatenate(datestamps),
            "y": np.concatenate(ys),
        })
        # Run forecast with the project's standard 9 quantile levels.
        out = self.model.forecast(
            df=df, h=horizon, freq=freq,
            quantiles=list(self.quantile_levels),
            presorted=False,
        )
        # Extract point (alias) and per-quantile columns per unique_id.
        alias = getattr(self.model, "alias", self.model_name)
        # Identify quantile columns: alias-q-{int(q*100)}
        q_cols = {}
        for q in self.quantile_levels:
            col = f"{alias}-q-{int(q*100)}"
            if col in out.columns:
                q_cols[q] = col
        # Group per unique_id.
        points, quantiles = [], []
        out_sorted = out.sort_values(["unique_id", "ds"])
        for uid, grp in out_sorted.groupby("unique_id", sort=True):
            point_arr = grp[alias].values[:horizon].astype(np.float32)
            qmat = np.zeros((len(self.quantile_levels), horizon), dtype=np.float32)
            for qi, q in enumerate(self.quantile_levels):
                col = q_cols.get(q)
                if col is not None:
                    qmat[qi] = grp[col].values[:horizon].astype(np.float32)
                else:
                    qmat[qi] = point_arr  # fall back to point if no quantile
            points.append(point_arr)
            quantiles.append(qmat)
        return points, quantiles


# Lazily-loaded dataset → (domain, no_daily) lookup used by FlowState.
# Vendored copy lives in gift_eval/data/ (gift_eval/_fm/fm_predictors.py → parents[1] = gift_eval/).
_DATASET_PROPS_PATH = Path(__file__).resolve().parents[1] / "data" / "dataset_properties.json"
_PRETTY_NAMES = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}
_DATASET_PROPS_CACHE: Dict[str, dict] | None = None


def _dataset_meta(dataset: str) -> Tuple[str | None, bool]:
    """Returns (domain, no_daily) for a dataset string like 'electricity/H'."""
    global _DATASET_PROPS_CACHE
    if _DATASET_PROPS_CACHE is None:
        with open(_DATASET_PROPS_PATH) as f:
            _DATASET_PROPS_CACHE = json.load(f)
    ds_key = dataset.split("/")[0].lower()
    ds_key = _PRETTY_NAMES.get(ds_key, ds_key)
    domain = _DATASET_PROPS_CACHE.get(ds_key, {}).get("domain")
    no_daily = "l2c" in dataset.lower()
    return domain, no_daily


