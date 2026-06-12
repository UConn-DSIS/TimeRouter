"""Base class and utilities for all forecasting models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import pandas as pd
import utilsforecast.processing as ufp
from gluonts.time_feature.seasonality import (
    DEFAULT_SEASONALITIES,
)
from gluonts.time_feature.seasonality import (
    get_seasonality as _get_seasonality,
)
from utilsforecast.processing import (
    backtest_splits,
    drop_index_if_pandas,
    join,
    maybe_compute_sort_indices,
    take_rows,
    vertical_concat,
)


def get_seasonality(
    freq: str,
    custom_seasonalities: dict[str, int] | None = None,
) -> int:
    """Get the seasonality of a frequency."""
    if custom_seasonalities is None:
        custom_seasonalities = dict()
    return _get_seasonality(
        freq,
        seasonalities=DEFAULT_SEASONALITIES | custom_seasonalities,
    )


def maybe_infer_freq(df: pd.DataFrame, freq: str | None) -> str:
    """Infer the frequency of the time series data."""
    if freq is not None:
        return freq
    sizes = df["unique_id"].value_counts(sort=True)
    times = df.loc[df["unique_id"] == sizes.index[0], "ds"].sort_values()
    if times.dt.tz is not None:
        times = times.dt.tz_convert("UTC").dt.tz_localize(None)
    inferred_freq = pd.infer_freq(times.values)
    if inferred_freq is None:
        raise RuntimeError(
            "Could not infer the frequency of the time column. This could be due "
            "to inconsistent intervals. Please check your data for missing, "
            "duplicated or irregular timestamps"
        )
    return inferred_freq


def maybe_convert_col_to_datetime(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    if not pd.api.types.is_datetime64_any_dtype(df[col_name]):
        df = df.copy()
        df[col_name] = pd.to_datetime(df[col_name])
    return df


class ForecastModel(ABC):
    """Abstract base class for all forecasting models."""

    @property
    @abstractmethod
    def alias(self) -> str:
        """Model name used in output DataFrame columns."""
        ...

    @staticmethod
    def _maybe_infer_freq(
        df: pd.DataFrame,
        freq: str | None,
    ) -> str:
        return maybe_infer_freq(df, freq)

    def _maybe_get_seasonality(self, freq: str) -> int:
        if hasattr(self, "season_length"):
            if self.season_length is not None:
                return self.season_length
            else:
                return get_seasonality(freq)
        else:
            return get_seasonality(freq)

    @abstractmethod
    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
        presorted: bool = False,
    ) -> pd.DataFrame:
        """Generate forecasts for time series data.

        Args:
            df: DataFrame with columns [unique_id, ds, y].
            h: Forecast horizon.
            freq: Frequency string. Inferred if None.
            level: Confidence levels for prediction intervals (e.g. [80, 95]).
            quantiles: Quantile levels (e.g. [0.1, 0.5, 0.9]).
            presorted: If True, df is already sorted by [unique_id, ds].

        Returns:
            DataFrame with columns [unique_id, ds, {alias}, ...].
        """
        ...

    def cross_validation(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None = None,
        n_windows: int = 1,
        step_size: int | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
    ) -> pd.DataFrame:
        """Expanding-window cross-validation using backtest_splits.

        Ported from the original Forecaster.cross_validation() to ensure
        identical window splitting and date handling.

        Returns:
            DataFrame with columns [unique_id, ds, cutoff, y, {alias}, ...].
        """
        freq = self._maybe_infer_freq(df, freq)
        df = maybe_convert_col_to_datetime(df, "ds")

        results = []
        sort_idxs = maybe_compute_sort_indices(df, "unique_id", "ds")
        if sort_idxs is not None:
            df = take_rows(df, sort_idxs)

        splits = backtest_splits(
            df,
            n_windows=n_windows,
            h=h,
            id_col="unique_id",
            time_col="ds",
            freq=pd.tseries.frequencies.to_offset(freq),
            step_size=h if step_size is None else step_size,
        )

        for _, (cutoffs, train, valid) in enumerate(splits):
            y_pred = self.forecast(
                df=train,
                h=h,
                freq=freq,
                level=level,
                quantiles=quantiles,
                presorted=True,
            )
            y_pred = join(y_pred, cutoffs, on="unique_id", how="left")
            result = join(
                valid[["unique_id", "ds", "y"]],
                y_pred,
                on=["unique_id", "ds"],
            )
            results.append(result)

        out = vertical_concat(results)
        out = drop_index_if_pandas(out)
        first_out_cols = ["unique_id", "ds", "cutoff", "y"]
        remaining_cols = [c for c in out.columns if c not in first_out_cols]
        return out[first_out_cols + remaining_cols]


class QuantileConverter:
    """Handles inputs and outputs for probabilistic forecasts."""

    def __init__(
        self,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
    ):
        level, quantiles, level_was_provided = self._prepare_level_and_quantiles(
            level, quantiles
        )
        self.level = level
        self.quantiles = quantiles
        self.level_was_provided = level_was_provided

    @staticmethod
    def _prepare_level_and_quantiles(
        level: list[int | float] | None,
        quantiles: list[float] | None,
    ) -> tuple[list[int | float] | None, list[float] | None, bool]:
        if level is not None and quantiles is not None:
            raise ValueError(
                "You must not provide both `level` and `quantiles` simultaneously."
            )
        if quantiles is None and level is not None:
            _quantiles = []
            for lv in level:
                q_lo, q_hi = QuantileConverter._level_to_quantiles(lv)
                _quantiles.append(q_lo)
                _quantiles.append(q_hi)
            quantiles = sorted(set(_quantiles))
            level_was_provided = True
            return level, quantiles, level_was_provided
        if level is None and quantiles is not None:
            if not all(0 < q < 1 for q in quantiles):
                raise ValueError("`quantiles` should be floats between 0 and 1.")
            level = [abs(int(100 - 200 * q)) for q in quantiles]
            level_was_provided = False
            return sorted(set(level)), quantiles, level_was_provided
        else:
            return None, None, False

    @staticmethod
    def _level_to_quantiles(level: int | float) -> tuple[float, float]:
        """Given a prediction interval level (e.g. 80) return lower & upper quantiles."""
        alpha = 1 - level / 100
        q_lo = alpha / 2
        q_hi = 1 - q_lo
        return q_lo, q_hi

    def maybe_convert_level_to_quantiles(
        self,
        df: pd.DataFrame,
        models: list[str],
    ) -> pd.DataFrame:
        """Convert level columns to quantile columns if level was provided."""
        if self.level_was_provided or self.level is None:
            return df
        if self.quantiles is None:
            raise ValueError("No quantiles were provided.")
        out_cols = [c for c in df.columns if "-lo-" not in c and "-hi-" not in c]
        df = ufp.copy_if_pandas(df, deep=False)
        for model in models:
            for q in sorted(self.quantiles):
                if q == 0.5:
                    col = model
                else:
                    lv = int(100 - 200 * q)
                    hi_or_lo = "lo" if lv > 0 else "hi"
                    lv = abs(lv)
                    col = f"{model}-{hi_or_lo}-{lv}"
                q_col = f"{model}-q-{int(q * 100)}"
                df = ufp.assign_columns(df, q_col, df[col])
                out_cols.append(q_col)
        return df[out_cols]

    def maybe_convert_quantiles_to_level(
        self,
        df: pd.DataFrame,
        models: list[str],
    ) -> pd.DataFrame:
        """Convert quantile columns to level columns if quantiles were provided."""
        if not self.level_was_provided or self.quantiles is None:
            return df
        if self.level is None:
            raise ValueError("No levels were provided.")
        out_cols = [c for c in df.columns if "-q-" not in c]
        df = ufp.copy_if_pandas(df, deep=False)
        for model in models:
            if 0 in self.level:
                mid_col = f"{model}-q-50"
                if mid_col in df:
                    df = ufp.assign_columns(df, model, df[mid_col])
                    if model not in out_cols:
                        out_cols.append(model)
            for lv in self.level:
                q_lo, q_hi = self._level_to_quantiles(lv)
                lo_src = f"{model}-q-{int(q_lo * 100)}"
                hi_src = f"{model}-q-{int(q_hi * 100)}"
                lo_tgt = f"{model}-lo-{lv}"
                hi_tgt = f"{model}-hi-{lv}"
                if lo_src in df and hi_src in df:
                    df = ufp.assign_columns(df, lo_tgt, df[lo_src])
                    df = ufp.assign_columns(df, hi_tgt, df[hi_src])
                    out_cols.extend([lo_tgt, hi_tgt])
        return df[out_cols]
