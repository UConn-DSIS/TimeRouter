"""Minimal vendored model registry (chronos only).

Vendored from ``timeagents.models`` for a self-contained gift_eval/ package.
Only the Chronos-2 wrapper is needed by the deployed K=4 pool — the other FMs
(FlowState, PatchTST-FM, Sundial) load via their own predictor classes in
``gift_eval/_fm/fm_predictors.py``.

Usage:
    from gift_eval._fm.chronos_backend import get_model
    model = get_model("chronos", repo_id="amazon/chronos-2")
    result_df = model.forecast(df, h=24, freq="H")
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from gift_eval._fm.chronos_backend.base import ForecastModel

logger = logging.getLogger(__name__)

# Quantile levels used throughout the system (must match old repo)
QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Per-model GPU batch sizes (tuned for 48GB A6000)
BATCH_SIZE_DEFAULTS: Dict[str, int] = {
    "chronos": 128,
}

# Lazy import registry: model_name -> (module_path, class_name)
_REGISTRY: Dict[str, tuple[str, str]] = {
    "chronos": ("gift_eval._fm.chronos_backend.chronos", "Chronos"),
}

# Class-level model instance cache (shared across calls)
_model_cache: Dict[str, ForecastModel] = {}


def get_model(name: str, **kwargs: Any) -> ForecastModel:
    """Get or create a model instance by name (with caching)."""
    name_lower = name.lower()
    cache_key = f"{name_lower}:{sorted(kwargs.items())}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]
    if name_lower not in _REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(_REGISTRY.keys())}")

    module_path, class_name = _REGISTRY[name_lower]
    import importlib
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    instance = cls(**kwargs)
    _model_cache[cache_key] = instance
    logger.info(f"Loaded model: {name}")
    return instance


def list_models() -> list[str]:
    """List all available model names."""
    return sorted(_REGISTRY.keys())
