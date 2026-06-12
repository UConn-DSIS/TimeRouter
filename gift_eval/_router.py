"""Load shipped 5-seed OvA ensemble and apply the deployed decision rule.

Decision rule (paper §3.4):
    margin    = top1_proba - top2_proba       (decision-space confidence)
    diversity = mean over horizon of std across z-normalized FM forecasts
    if margin < tau_m  OR  diversity < tau_d:   defer to ensemble fallback (CV-inverse-weighted average)
    else:                                        argmax over the K-class softmax → pick that FM

Deployed (XGB) thresholds: (tau_m, tau_d) = (0.07, 0.07).
v4 RF head thresholds:     (tau_m, tau_d) = (0.05, 0.02).
"""
from __future__ import annotations

import glob
import json
import pickle
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import xgboost as xgb

from ._features import POOL


TAU_M = 0.07
TAU_D = 0.07
SEEDS = [42, 43, 44, 45, 46]

# Public HuggingFace repo hosting the shipped 5-seed OvA checkpoints. Used as the
# download source when no usable local checkpoint dir is given (see resolve_ckpt_dir).
DEFAULT_CKPT_REPO = "nkh/timerouter-v1"


def _is_local_ckpt(p: Path) -> bool:
    """True if ``p`` is a directory already holding loadable router weights."""
    return p.is_dir() and (bool(list(p.glob("seed*.json"))) or (p / "rf_ensembles.pkl").exists())


def resolve_ckpt_dir(spec: str | Path, hf_repo: str = DEFAULT_CKPT_REPO) -> Path:
    """Resolve a checkpoint location to a local directory, downloading from HF if needed.

    ``spec`` may be:
      * a local directory containing ``seed*.json`` / ``rf_ensembles.pkl`` -> used as-is;
      * a HuggingFace repo id like ``"nkh/timerouter-v1"`` (``org/name``, not a local
        path) -> fetched via ``snapshot_download`` and the cache dir returned;
      * anything else (e.g. a missing local default path) -> falls back to downloading
        ``hf_repo``.

    This lets reproduction work from a clone that does not ship the ~170 MB checkpoint:
    pass ``--ckpt-dir nkh/timerouter-v1`` explicitly, or just let the missing-local-dir
    fallback pull ``hf_repo`` automatically.
    """
    p = Path(spec)
    if _is_local_ckpt(p):
        return p
    s = str(spec)
    looks_like_repo_id = (s.count("/") == 1 and not p.is_absolute()
                          and not s.startswith(".") and not p.exists())
    repo_id = s if looks_like_repo_id else hf_repo
    from huggingface_hub import snapshot_download
    local = snapshot_download(repo_id=repo_id, repo_type="model")
    return Path(local)


def load_ensemble(ckpt_dir: str | Path) -> List[List]:
    """Load the 5-seed OvA ensemble.

    Auto-detects format:
      - ``ckpt_dir/seed*.json``    : XGB JSON checkpoint (deployed default)
      - ``ckpt_dir/rf_ensembles.pkl``: sklearn RF ensemble pickle (v4 RF head)
    Returns a list (len=5) of lists (len=K=3) of fitted classifiers.
    """
    ckpt_dir = Path(ckpt_dir)
    pkl_path = ckpt_dir / "rf_ensembles.pkl"
    if pkl_path.exists():
        # sklearn RF pickle (head-ablation v4 format)
        payload = pickle.load(open(pkl_path, "rb"))
        return payload["ensembles"]

    # XGB JSON ckpts
    paths = sorted(glob.glob(str(ckpt_dir / "seed*.json")))
    if not paths:
        raise FileNotFoundError(f"No seed*.json or rf_ensembles.pkl under {ckpt_dir}")
    ensembles: List[List[xgb.XGBClassifier]] = []
    for p in paths:
        art = json.load(open(p))
        per_class: List[xgb.XGBClassifier] = []
        for pc in art["per_class"]:
            m = xgb.XGBClassifier()
            m.load_model(bytearray(json.dumps(pc["xgb_model"]).encode()))
            per_class.append(m)
        ensembles.append(per_class)
    return ensembles


def predict_proba_ova(ensembles: List[List[xgb.XGBClassifier]],
                      X: np.ndarray, K: int) -> np.ndarray:
    """OvA → softmax average across seeds. Returns (N, K)."""
    out = np.zeros((len(X), K), dtype=np.float32)
    for ens in ensembles:
        scores = np.zeros((len(X), K), dtype=np.float32)
        for c, m in enumerate(ens):
            scores[:, c] = m.predict_proba(X)[:, 1]
        scores /= scores.sum(axis=1, keepdims=True) + 1e-9
        out += scores
    return out / len(ensembles)


def cv_inverse_weighted(forecasts: Dict[str, np.ndarray],
                        cv_scores: Dict[str, float]) -> np.ndarray:
    """CV-inverse-weighted average over the pool's forecasts (matches deployed Ens)."""
    inv: Dict[str, float] = {}
    for m, fc in forecasts.items():
        s = cv_scores.get(m)
        if s is not None and np.isfinite(s):
            inv[m] = 1.0 / (s + 1e-8)
    if not inv:
        return np.mean(list(forecasts.values()), axis=0)
    total = sum(inv.values())
    out = np.zeros_like(next(iter(forecasts.values())), dtype=np.float64)
    for m, w in inv.items():
        out += (w / total) * np.asarray(forecasts[m], dtype=np.float64)
    return out


def cv_inverse_weighted_quantiles(quantile_forecasts: Dict[str, np.ndarray],
                                  cv_scores: Dict[str, float]) -> np.ndarray:
    """Same weighting applied independently to each of the 9 quantile bands.

    Inputs:
        quantile_forecasts: {fm: array shape (9, horizon)}
    Output:
        array shape (9, horizon)
    """
    inv: Dict[str, float] = {}
    for m, q in quantile_forecasts.items():
        s = cv_scores.get(m)
        if s is not None and np.isfinite(s) and q is not None:
            inv[m] = 1.0 / (s + 1e-8)
    if not inv:
        return np.mean(np.stack([q for q in quantile_forecasts.values() if q is not None]),
                       axis=0)
    total = sum(inv.values())
    first_q = np.asarray(quantile_forecasts[next(iter(inv))], dtype=np.float64)
    out = np.zeros_like(first_q)
    for m, w in inv.items():
        q = np.asarray(quantile_forecasts[m], dtype=np.float64)
        out += (w / total) * q
    return out


def decide(probas: np.ndarray, diversity: np.ndarray,
           tau_m: float = TAU_M, tau_d: float = TAU_D) -> Dict[str, np.ndarray]:
    """Apply the combined OR gate. Returns dict with picks, gated, margin."""
    sorted_p = np.sort(probas, axis=1)
    margin = sorted_p[:, -1] - sorted_p[:, -2]
    picks = probas.argmax(axis=1)
    gated = (margin < tau_m) | (diversity < tau_d)
    return {"picks": picks, "gated": gated, "margin": margin}
