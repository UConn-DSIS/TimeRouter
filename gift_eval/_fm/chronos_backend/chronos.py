"""Chronos foundation model wrapper."""

from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
from chronos import (
    BaseChronosPipeline,
    Chronos2Pipeline,
    ChronosBoltPipeline,
    ChronosPipeline,
)
from tqdm import tqdm

from gift_eval._fm.chronos_backend.base import ForecastModel, QuantileConverter
from gift_eval._fm.chronos_backend.utils import TimeSeriesDataset


class Chronos(ForecastModel):
    """Chronos models for time series forecasting.

    See https://github.com/amazon-science/chronos-forecasting
    """

    # Class-level cache for models (shared across instances)
    _model_cache: dict = {}

    def __init__(
        self,
        repo_id: str = "amazon/chronos-2",
        batch_size: int = 1024,
        alias: str = "Chronos",
        # Default to False — our inference batches mix multiple rolling windows
        # of the same series, so jointly=True would leak future info via
        # in-context learning. Official notebook uses jointly=True but only
        # after manually separating entries by window_idx; we don't do that.
        predict_batches_jointly: bool = False,
    ):
        self.repo_id = repo_id
        self.batch_size = batch_size
        self._alias = alias
        self.predict_batches_jointly = predict_batches_jointly

    @property
    def alias(self) -> str:
        return self._alias

    def _get_or_create_model(self) -> BaseChronosPipeline:
        """Get cached model or create a new one."""
        if self.repo_id in Chronos._model_cache:
            return Chronos._model_cache[self.repo_id]

        device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = BaseChronosPipeline.from_pretrained(
            self.repo_id,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
        )
        # Chronos-2 / Bolt pipelines never call .eval() themselves, so the
        # underlying nn.Module stays in training mode and dropout layers
        # randomise every forward pass. Force eval() for deterministic inference.
        inner = getattr(model, "model", None)
        if inner is not None and hasattr(inner, "eval"):
            inner.eval()
        Chronos._model_cache[self.repo_id] = model
        return model

    @contextmanager
    def _get_model(self) -> BaseChronosPipeline:
        """Context manager for backward compatibility - now uses caching."""
        model = self._get_or_create_model()
        try:
            yield model
        finally:
            pass

    @classmethod
    def clear_cache(cls):
        """Clear the model cache and free GPU memory."""
        for key in list(cls._model_cache.keys()):
            del cls._model_cache[key]
        cls._model_cache.clear()
        torch.cuda.empty_cache()

    # Quantiles supported by Chronos-Bolt and Chronos-2 models
    BOLT_SUPPORTED_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    def _adjust_quantiles_for_bolt(
        self,
        model: BaseChronosPipeline,
        quantiles: list[float] | None,
    ) -> list[float] | None:
        """Adjust quantiles for Chronos-Bolt/Chronos-2 compatibility."""
        if quantiles is None:
            return None

        if not isinstance(model, (ChronosBoltPipeline, Chronos2Pipeline)):
            return quantiles

        adjusted = []
        for q in quantiles:
            if q in self.BOLT_SUPPORTED_QUANTILES:
                adjusted.append(q)
            else:
                nearest = min(
                    self.BOLT_SUPPORTED_QUANTILES,
                    key=lambda x: abs(x - q)
                )
                adjusted.append(nearest)

        seen = set()
        unique_adjusted = []
        for q in adjusted:
            if q not in seen:
                seen.add(q)
                unique_adjusted.append(q)

        return unique_adjusted

    def _predict(
        self,
        model: BaseChronosPipeline,
        dataset: TimeSeriesDataset,
        h: int,
        quantiles: list[float] | None,
    ) -> tuple[np.ndarray, np.ndarray | None, list[float] | None]:
        """Handles distinction between predict and predict_quantiles."""
        adjusted_quantiles = self._adjust_quantiles_for_bolt(model, quantiles)

        # Chronos-2 supports cross-series in-context learning via
        # predict_batches_jointly. Other Chronos variants don't accept this
        # kwarg so we only pass it when the loaded pipeline is Chronos2.
        is_chronos2 = isinstance(model, Chronos2Pipeline)
        extra_kwargs = (
            {"predict_batches_jointly": self.predict_batches_jointly}
            if is_chronos2 else {}
        )

        if adjusted_quantiles is not None:
            fcsts = [
                model.predict_quantiles(
                    batch,
                    prediction_length=h,
                    quantile_levels=adjusted_quantiles,
                    **extra_kwargs,
                )
                for batch in tqdm(dataset, disable=True)
            ]
            fcsts_quantiles, fcsts_mean = zip(*fcsts, strict=False)
            if is_chronos2:
                fcsts_mean = [f_mean for fcst in fcsts_mean for f_mean in fcst]
                fcsts_quantiles = [
                    f_quantile
                    for fcst in fcsts_quantiles
                    for f_quantile in fcst
                ]
            fcsts_mean_np = torch.cat(fcsts_mean).numpy()
            fcsts_quantiles_np = torch.cat(fcsts_quantiles).numpy()
            return fcsts_mean_np, fcsts_quantiles_np, adjusted_quantiles
        else:
            fcsts = [
                model.predict(
                    batch,
                    prediction_length=h,
                    **extra_kwargs,
                )
                for batch in tqdm(dataset, disable=True)
            ]
            if isinstance(model, Chronos2Pipeline):
                fcsts = [f_fcst for fcst in fcsts for f_fcst in fcst]
            fcsts = torch.cat(fcsts)
            if isinstance(model, ChronosPipeline):
                fcsts_mean = fcsts.mean(dim=1)
            elif isinstance(model, ChronosBoltPipeline | Chronos2Pipeline):
                fcsts_mean = fcsts[:, model.quantiles.index(0.5), :]
            else:
                raise ValueError(f"Unsupported model: {self.repo_id}")
            fcsts_mean_np = fcsts_mean.numpy()
            fcsts_quantiles_np = None
        return fcsts_mean_np, fcsts_quantiles_np, adjusted_quantiles

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
        presorted: bool = False,
    ) -> pd.DataFrame:
        freq = self._maybe_infer_freq(df, freq)
        qc = QuantileConverter(level=level, quantiles=quantiles)
        dataset = TimeSeriesDataset.from_df(df, batch_size=self.batch_size, presorted=presorted)
        fcst_df = dataset.make_future_dataframe(h=h, freq=freq)
        with self._get_model() as model:
            fcsts_mean_np, fcsts_quantiles_np, actual_quantiles = self._predict(
                model,
                dataset,
                h,
                quantiles=qc.quantiles,
            )
        fcst_df[self._alias] = fcsts_mean_np.reshape(-1, 1)
        if actual_quantiles is not None and fcsts_quantiles_np is not None:
            for i, q in enumerate(actual_quantiles):
                fcst_df[f"{self._alias}-q-{int(q * 100)}"] = fcsts_quantiles_np[
                    ..., i
                ].reshape(-1, 1)
            qc.quantiles = actual_quantiles
            fcst_df = qc.maybe_convert_quantiles_to_level(
                fcst_df,
                models=[self._alias],
            )
        return fcst_df
