"""Linear frozen-teacher BM-JEPA wrapper."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models.rrr import mse, predict, reduced_rank_fit


@dataclass
class LinearBMJEPA:
    """Rank-constrained linear BM-JEPA predictor."""

    bottleneck_dim: int
    ridge: float = 1e-8
    coefficient_: np.ndarray | None = None

    def fit(self, context: np.ndarray, stacked_targets: np.ndarray) -> "LinearBMJEPA":
        self.coefficient_ = reduced_rank_fit(
            context, stacked_targets, rank=self.bottleneck_dim, ridge=self.ridge
        )
        return self

    def predict(self, context: np.ndarray) -> np.ndarray:
        if self.coefficient_ is None:
            raise RuntimeError("LinearBMJEPA must be fit before predict")
        return predict(context, self.coefficient_)

    def score_mse(self, context: np.ndarray, stacked_targets: np.ndarray) -> float:
        return mse(stacked_targets, self.predict(context))

