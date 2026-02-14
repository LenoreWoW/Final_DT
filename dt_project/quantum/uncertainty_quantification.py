"""Uncertainty quantification stub."""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class UncertaintyResult:
    mean: float = 0.0
    std: float = 0.1
    confidence_interval: tuple = (0.0, 0.0)
    credible_interval: tuple = (0.0, 0.0)


class UncertaintyQuantifier:
    def __init__(self, config=None):
        self._config = config or {}

    def quantify(self, data):
        arr = np.array(data) if not isinstance(data, np.ndarray) else data
        return UncertaintyResult(
            mean=float(np.mean(arr)),
            std=float(np.std(arr)),
            confidence_interval=(float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))),
            credible_interval=(float(np.percentile(arr, 5)), float(np.percentile(arr, 95))),
        )

    def propagate_uncertainty(self, model_fn, inputs, uncertainties):
        return UncertaintyResult()


class BayesianUncertainty:
    def __init__(self):
        pass

    def estimate(self, data):
        return UncertaintyResult(mean=float(np.mean(data)), std=float(np.std(data)))


class MonteCarloUncertainty:
    def __init__(self, n_samples=1000):
        self.n_samples = n_samples

    def estimate(self, model_fn, params):
        return UncertaintyResult()
