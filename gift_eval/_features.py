"""Inline 270-dim feature builder for TimeRouter (deployed pool K=3).

This replicates exactly what `final_scripts/pipeline/_pool.py:feature_order(pool)`
produces when ``pool = ["chronos", "patchtst_fm", "sundial"]``, but consumes
arrays directly (no parquet round-trip).

Per-row inputs:
  context    : np.ndarray, the input series (any length)
  horizon    : int,        prediction horizon
  freq       : str,        gluonts-style freq (e.g. "H", "5T", "D")
  forecasts  : dict[fm -> np.ndarray of shape (horizon,)]   future-horizon point forecasts
  cv_scores  : dict[fm -> float]                            per-FM CV-backtest MASE

Returns a (270,) float32 array in the deployed feature order.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

POOL: List[str] = ['chronos', 'flowstate', 'patchtst_fm', 'sundial']
PRED_LEN = 32
RAW_SERIES_LEN = 128

TS_STATS_COLS = [
    "feat_mean", "feat_std", "feat_cv", "feat_min", "feat_max",
    "feat_range", "feat_iqr", "feat_skewness", "feat_kurtosis",
    "feat_acf_lag1", "feat_acf_lag5", "feat_acf_lag10",
    "feat_trend_slope", "feat_diff_std", "feat_diff_mean_abs",
    "feat_zero_crossings", "feat_turning_points", "feat_log_length",
]
CTX_STATIC_COLS = ["ctx_horizon", "ctx_series_length",
                   "ctx_horizon_to_length", "ctx_freq_numeric"]
REGIME_SHIFT_COLS = [
    "rs_mean_shift", "rs_std_ratio", "rs_trend_change", "rs_acf1_change",
    "rs_level_shift", "rs_ks_stat", "rs_max_drawdown_ratio", "rs_energy_ratio",
]
RAW_SERIES_COLS = [f"raw_{i}" for i in range(RAW_SERIES_LEN)]

FREQ_MAP = {
    "S": 1, "10S": 10, "T": 60, "5T": 300, "10T": 600, "15T": 900,
    "30T": 1800, "H": 3600, "6H": 21600, "D": 86400, "W": 604800,
    "M": 2592000, "MS": 2592000, "Q": 7776000, "QS": 7776000,
    "Y": 31536000, "YS": 31536000, "A": 31536000,
}


def feature_order(pool: Sequence[str] = POOL) -> List[str]:
    """Deployed 270-col feature order."""
    cols: List[str] = []
    cols += TS_STATS_COLS                                              # 18
    cols += [f"cv_score_{m}" for m in pool]                            # 3K
    cols += [f"cv_rank_{m}"  for m in pool]                            # 3K
    cols += [f"cv_gap_{m}"   for m in pool]                            # 3K
    cols += ["cv_gap_top2", "cv_score_std", "cv_score_range",
             "cv_n_survivors", "cv_score_entropy", "cv_best_confidence"]  # 6
    cols += CTX_STATIC_COLS + ["ctx_n_available_models"]               # 5
    cols += REGIME_SHIFT_COLS                                          # 8
    cols += RAW_SERIES_COLS                                            # 128
    for m in pool:
        cols += [f"{m}_pred_{i}" for i in range(PRED_LEN)]             # 32K
    return cols


# ---------------------------------------------------------------------------
#  Per-row feature blocks (mirror routing_learning/features.py)
# ---------------------------------------------------------------------------
def _ts_stats(context: np.ndarray) -> Dict[str, float]:
    feats: Dict[str, float] = {k: 0.0 for k in TS_STATS_COLS}
    y = np.asarray(context, dtype=np.float64)
    y_clean = y[np.isfinite(y)]
    if len(y_clean) < 3:
        feats["feat_log_length"] = float(np.log1p(len(context)))
        return feats

    mean = float(np.mean(y_clean))
    std = float(np.std(y_clean, ddof=1)) if len(y_clean) > 1 else 0.0
    feats["feat_mean"] = mean
    feats["feat_std"] = std
    feats["feat_cv"] = std / max(abs(mean), 1e-8)
    feats["feat_min"] = float(np.min(y_clean))
    feats["feat_max"] = float(np.max(y_clean))
    feats["feat_range"] = feats["feat_max"] - feats["feat_min"]
    feats["feat_iqr"] = float(np.percentile(y_clean, 75) - np.percentile(y_clean, 25))

    s_norm = max(std, 1e-8)
    feats["feat_skewness"] = float(np.mean(((y_clean - mean) / s_norm) ** 3))
    feats["feat_kurtosis"] = float(np.mean(((y_clean - mean) / s_norm) ** 4) - 3)

    y_c = y_clean - np.mean(y_clean)
    var = float(np.var(y_clean))
    for lag in (1, 5, 10):
        if lag < len(y_c) and var > 1e-12:
            feats[f"feat_acf_lag{lag}"] = float(np.clip(
                np.mean(y_c[lag:] * y_c[:-lag]) / var, -1, 1))

    x = np.arange(len(y_clean), dtype=np.float64)
    xc = x - x.mean()
    denom = float(np.sum(xc ** 2))
    if denom > 1e-12:
        feats["feat_trend_slope"] = float(
            np.sum(xc * (y_clean - y_clean.mean())) / denom / s_norm)

    diff1 = np.diff(y_clean)
    if len(diff1) > 0:
        feats["feat_diff_std"] = float(np.std(diff1))
        feats["feat_diff_mean_abs"] = float(np.mean(np.abs(diff1)))

    signs = np.sign(y_c)
    feats["feat_zero_crossings"] = float(
        np.sum(signs[1:] != signs[:-1]) / max(len(y_c) - 1, 1))

    if len(y_clean) > 2:
        d = np.diff(y_clean)
        feats["feat_turning_points"] = float(
            np.sum(d[1:] * d[:-1] < 0) / max(len(y_clean) - 2, 1))

    feats["feat_log_length"] = float(np.log1p(len(context)))
    return feats


def _ctx_static(horizon: int, series_length: int, freq: str) -> Dict[str, float]:
    f = freq.upper().rstrip("S") if freq.upper() not in FREQ_MAP else freq.upper()
    sec = FREQ_MAP.get(f, FREQ_MAP.get(freq.upper(), 86400))
    return {
        "ctx_horizon": float(horizon),
        "ctx_series_length": float(series_length),
        "ctx_horizon_to_length": float(horizon) / max(series_length, 1),
        "ctx_freq_numeric": float(np.log1p(sec)),
    }


def _regime_shift(context: np.ndarray) -> Dict[str, float]:
    keys = REGIME_SHIFT_COLS
    feats = {k: 0.0 for k in keys}
    y = np.asarray(context, dtype=np.float64)
    valid = y[np.isfinite(y)]
    n = len(valid)
    if n < 10:
        return feats

    mid = n // 2
    h1, h2 = valid[:mid], valid[mid:]
    global_std = max(float(np.std(valid)), 1e-8)
    feats["rs_mean_shift"] = abs(float(np.mean(h2)) - float(np.mean(h1))) / global_std

    std1 = max(float(np.std(h1)), 1e-8)
    std2 = max(float(np.std(h2)), 1e-8)
    feats["rs_std_ratio"] = std2 / std1

    def _slope(arr):
        x = np.arange(len(arr), dtype=np.float64)
        xc = x - x.mean()
        d = float(np.sum(xc ** 2))
        return float(np.sum(xc * (arr - arr.mean())) / d) if d > 1e-12 else 0.0

    feats["rs_trend_change"] = abs(_slope(h2) - _slope(h1)) / global_std

    def _acf1(arr):
        if len(arr) < 3:
            return 0.0
        c = arr - np.mean(arr)
        v = float(np.var(arr))
        return float(np.clip(np.mean(c[1:] * c[:-1]) / v, -1, 1)) if v > 1e-12 else 0.0

    feats["rs_acf1_change"] = abs(_acf1(h2) - _acf1(h1))

    n_segments = min(8, n // 4)
    if n_segments >= 2:
        seg_len = n // n_segments
        seg_means = [float(np.mean(valid[i*seg_len:(i+1)*seg_len])) for i in range(n_segments)]
        feats["rs_level_shift"] = max(
            abs(seg_means[i+1] - seg_means[i]) for i in range(len(seg_means)-1)
        ) / global_std

    try:
        from scipy.stats import ks_2samp
        feats["rs_ks_stat"] = float(ks_2samp(h1, h2).statistic)
    except Exception:
        pass

    def _max_dd(arr):
        peak = arr[0]
        dd = 0.0
        for v in arr:
            if v > peak: peak = v
            if peak - v > dd: dd = peak - v
        return float(dd)

    dd1, dd2 = _max_dd(h1), _max_dd(h2)
    feats["rs_max_drawdown_ratio"] = dd2 / max(dd1, 1e-8)

    e1 = float(np.sum(np.diff(h1) ** 2)) / max(len(h1) - 1, 1)
    e2 = float(np.sum(np.diff(h2) ** 2)) / max(len(h2) - 1, 1)
    feats["rs_energy_ratio"] = e2 / max(e1, 1e-8)
    return feats


def _resample_to_fixed(arr: np.ndarray, target_len: int) -> np.ndarray:
    n = len(arr)
    if n == 0:
        return np.zeros(target_len, dtype=np.float64)
    if n == target_len:
        return np.asarray(arr, dtype=np.float64)
    if n < target_len:
        out = np.zeros(target_len, dtype=np.float64)
        out[target_len - n:] = arr
        return out
    x_old = np.linspace(0, 1, n)
    x_new = np.linspace(0, 1, target_len)
    return np.interp(x_new, x_old, arr)


def _raw_series(context: np.ndarray) -> Dict[str, float]:
    valid = np.asarray(context, dtype=np.float64)
    valid = valid[np.isfinite(valid)]
    if len(valid) < 2:
        return {k: 0.0 for k in RAW_SERIES_COLS}
    mean = float(np.mean(valid))
    std = max(float(np.std(valid)), 1e-8)
    normed = (np.asarray(context, dtype=np.float64) - mean) / std
    normed = np.where(np.isfinite(normed), normed, 0.0)
    rs = _resample_to_fixed(normed, RAW_SERIES_LEN)
    return {f"raw_{i}": float(rs[i]) for i in range(RAW_SERIES_LEN)}


def _model_preds(context: np.ndarray,
                 forecasts: Dict[str, np.ndarray],
                 pool: Sequence[str] = POOL) -> Dict[str, float]:
    valid = np.asarray(context, dtype=np.float64)
    valid = valid[np.isfinite(valid)]
    mean = float(np.mean(valid)) if len(valid) else 0.0
    std = max(float(np.std(valid)), 1e-8) if len(valid) else 1.0
    feats: Dict[str, float] = {}
    for m in pool:
        f = forecasts.get(m)
        if f is None or len(f) == 0 or not np.all(np.isfinite(f)):
            rs = np.zeros(PRED_LEN)
        else:
            rs = _resample_to_fixed((np.asarray(f) - mean) / std, PRED_LEN)
        for i in range(PRED_LEN):
            feats[f"{m}_pred_{i}"] = float(rs[i])
    return feats


def _cv_block(cv_scores: Dict[str, float],
              pool: Sequence[str] = POOL) -> Dict[str, float]:
    """cv_score_{m} + cv_rank_{m} + cv_gap_{m} + 6 pool aggregates + ctx_n_available_models."""
    feats: Dict[str, float] = {}
    arr = np.array([cv_scores.get(m, np.nan) for m in pool], dtype=np.float64)
    finite = np.isfinite(arr)
    K = len(pool)

    # cv_score_{m}: raw
    for k, m in enumerate(pool):
        feats[f"cv_score_{m}"] = float(arr[k]) if finite[k] else float("nan")

    # rank: 0 = best, K = sentinel for NaN
    arr_for_sort = np.where(finite, arr, np.inf)
    order = np.argsort(arr_for_sort)
    rank = np.empty(K, dtype=np.float64)
    rank[order] = np.arange(K, dtype=np.float64)
    rank = np.where(finite, rank, float(K))
    for k, m in enumerate(pool):
        feats[f"cv_rank_{m}"] = float(rank[k])

    # gap to best
    if finite.any():
        best = float(arr_for_sort.min())
        for k, m in enumerate(pool):
            if finite[k]:
                feats[f"cv_gap_{m}"] = (float(arr[k]) - best) / max(abs(best), 1e-8)
            else:
                feats[f"cv_gap_{m}"] = float("nan")
    else:
        for m in pool:
            feats[f"cv_gap_{m}"] = float("nan")

    # 6 aggregates
    n_valid = int(finite.sum())
    has_top2 = n_valid >= 2
    if has_top2:
        sorted_vals = np.sort(arr_for_sort)[:n_valid]
        valid_arr = arr[finite]
        feats["cv_gap_top2"] = float(sorted_vals[1] - sorted_vals[0])
        feats["cv_score_std"] = float(np.std(valid_arr))
        feats["cv_score_range"] = float(valid_arr.max() - valid_arr.min())
        feats["cv_n_survivors"] = float(n_valid)
        inv = 1.0 / (valid_arr + 1e-8)
        probs = inv / inv.sum()
        feats["cv_score_entropy"] = float(-np.sum(probs * np.log(probs + 1e-12)))
        score_range = feats["cv_score_range"]
        feats["cv_best_confidence"] = (
            feats["cv_gap_top2"] / max(score_range, 1e-8) if score_range > 0 else 1.0
        )
    else:
        feats["cv_gap_top2"] = 0.0
        feats["cv_score_std"] = 0.0
        feats["cv_score_range"] = 0.0
        feats["cv_n_survivors"] = float(n_valid)
        feats["cv_score_entropy"] = 0.0
        feats["cv_best_confidence"] = 1.0

    feats["ctx_n_available_models"] = float(n_valid)
    return feats


def build_feature_vector(context: np.ndarray,
                         horizon: int,
                         freq: str,
                         forecasts: Dict[str, np.ndarray],
                         cv_scores: Dict[str, float],
                         pool: Sequence[str] = POOL) -> np.ndarray:
    """Return the (270,) float32 feature vector in deployed order."""
    feats: Dict[str, float] = {}
    feats.update(_ts_stats(context))
    feats.update(_cv_block(cv_scores, pool))
    feats.update(_ctx_static(horizon, len(context), freq))
    feats.update(_regime_shift(context))
    feats.update(_raw_series(context))
    feats.update(_model_preds(context, forecasts, pool))

    cols = feature_order(pool)
    vec = np.array([feats.get(c, 0.0) for c in cols], dtype=np.float32)
    return np.where(np.isfinite(vec), vec, 0.0).astype(np.float32)


# ---------------------------------------------------------------------------
#  Diversity (gate signal, computed across forecast outputs)
# ---------------------------------------------------------------------------
DIVERSITY_DOWNSAMPLE_MAX = 40


def compute_diversity(context: np.ndarray,
                      forecasts: Dict[str, np.ndarray]) -> float:
    """Per-row diversity = mean over horizon of std across z-normalized forecasts.

    Returns 0.0 if fewer than 2 valid forecasts.
    """
    ctx = np.asarray(context, dtype=np.float64)
    valid = ctx[np.isfinite(ctx)]
    if len(valid) == 0:
        return 0.0
    mean = float(valid.mean())
    std = max(float(valid.std()), 1e-5)
    preds = []
    for fc in forecasts.values():
        if fc is None:
            continue
        arr = np.asarray(fc, dtype=np.float64)
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            continue
        preds.append((arr - mean) / std)
    if len(preds) < 2:
        return 0.0
    p = np.stack(preds, axis=0)
    if p.shape[1] > DIVERSITY_DOWNSAMPLE_MAX:
        step = p.shape[1] // DIVERSITY_DOWNSAMPLE_MAX
        p = p[:, ::step]
    return float(np.std(p, axis=0).mean())
