"""Shared utilities for foundation model wrappers."""

from collections.abc import Iterable

import pandas as pd
import torch
from utilsforecast.processing import make_future_dataframe


class TimeSeriesDataset:
    def __init__(
        self,
        data: torch.Tensor,
        uids: Iterable,
        last_times: Iterable,
        batch_size: int,
    ):
        self.data = data
        self.uids = uids
        self.last_times = last_times
        self.batch_size = batch_size
        self.n_batches = len(data) // self.batch_size + (
            0 if len(data) % self.batch_size == 0 else 1
        )
        self.current_batch = 0

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        batch_size: int,
        dtype: torch.dtype = torch.bfloat16,
        presorted: bool = False,
    ):
        if not presorted:
            df = df.sort_values(by=["unique_id", "ds"])
        tensors = []
        uids = []
        last_times = []
        for uid, group in df.groupby("unique_id", sort=False):
            tensors.append(torch.tensor(group["y"].values, dtype=dtype))
            uids.append(uid)
            last_times.append(group["ds"].iloc[-1])
        return cls(tensors, uids, pd.Series(last_times), batch_size)

    def __len__(self):
        return self.n_batches

    def make_future_dataframe(self, h: int, freq: str) -> pd.DataFrame:
        return make_future_dataframe(
            uids=self.uids,
            last_times=pd.to_datetime(self.last_times),
            h=h,
            freq=freq,
        )  # type: ignore

    def __iter__(self):
        self.current_batch = 0  # Reset for new iteration
        return self

    def __next__(self):
        if self.current_batch < self.n_batches:
            start_idx = self.current_batch * self.batch_size
            end_idx = start_idx + self.batch_size
            self.current_batch += 1
            return self.data[start_idx:end_idx]
        else:
            raise StopIteration
