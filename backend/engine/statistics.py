"""
Statistical validation for quantum vs classical benchmark comparisons.

Provides paired t-tests with Bonferroni correction, Cohen's d effect sizes,
and confidence intervals for rigorous quantum advantage claims.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from scipy import stats


@dataclass
class StatisticalResult:
    """Result of a paired statistical comparison."""
    t_statistic: float
    p_value: float
    p_value_corrected: float  # Bonferroni-corrected
    cohens_d: float
    ci_lower: float
    ci_upper: float
    mean_quantum: float
    mean_classical: float
    std_quantum: float
    std_classical: float
    n_comparisons: int
    significant: bool  # After Bonferroni correction

    def to_dict(self):
        return {
            "t_statistic": float(self.t_statistic),
            "p_value": float(self.p_value),
            "p_value_corrected": float(self.p_value_corrected),
            "cohens_d": float(self.cohens_d),
            "ci_lower": float(self.ci_lower),
            "ci_upper": float(self.ci_upper),
            "mean_quantum": float(self.mean_quantum),
            "mean_classical": float(self.mean_classical),
            "std_quantum": float(self.std_quantum),
            "std_classical": float(self.std_classical),
            "n_comparisons": self.n_comparisons,
            "significant": self.significant,
        }


def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d for paired samples: d = mean(diffs) / std(diffs, ddof=1)."""
    diffs = np.asarray(group1, dtype=float) - np.asarray(group2, dtype=float)
    mean_diff = float(np.mean(diffs))
    std_diffs = float(np.std(diffs, ddof=1))
    if std_diffs == 0:
        # Zero variance: no effect if mean is also zero, otherwise cap at ±100
        # (Cohen's d > 10 already indicates zero distribution overlap;
        #  100 is a reasonable sentinel for JSON serialization)
        if mean_diff == 0:
            return 0.0
        return 100.0 if mean_diff > 0 else -100.0
    return mean_diff / std_diffs


def compute_cohens_d_independent(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size for two independent groups (pooled SD)."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


def compute_confidence_interval(
    data: np.ndarray, confidence: float = 0.95
) -> tuple:
    """Compute confidence interval for the mean."""
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return float(mean - h), float(mean + h)


def paired_comparison(
    quantum_vals: List[float],
    classical_vals: List[float],
    n_comparisons: int = 6,
) -> StatisticalResult:
    """
    Perform paired statistical comparison of quantum vs classical results.

    Args:
        quantum_vals: List of quantum metric values (one per run).
        classical_vals: List of classical metric values (one per run).
        n_comparisons: Number of simultaneous comparisons for Bonferroni correction.

    Returns:
        StatisticalResult with t-test, Cohen's d, corrected p-value, and CI.
    """
    q = np.array(quantum_vals, dtype=float)
    c = np.array(classical_vals, dtype=float)
    assert len(q) == len(c), "Quantum and classical arrays must have same length"

    # Paired t-test
    t_stat, p_val = stats.ttest_rel(q, c)

    # Clamp infinite t-statistic (scipy returns inf for zero-variance diffs)
    if not np.isfinite(t_stat):
        t_stat = np.sign(t_stat) * 1e6 if t_stat != 0 else 0.0

    # Bonferroni correction
    p_corrected = min(p_val * n_comparisons, 1.0)

    # Cohen's d (capped at ±100 for JSON serialization sanity)
    d = compute_cohens_d(q, c)
    d = max(-100.0, min(100.0, d))

    # 95% CI on the difference
    diffs = q - c
    ci_lower, ci_upper = compute_confidence_interval(diffs)

    return StatisticalResult(
        t_statistic=float(t_stat),
        p_value=float(p_val),
        p_value_corrected=float(p_corrected),
        cohens_d=d,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        mean_quantum=float(np.mean(q)),
        mean_classical=float(np.mean(c)),
        std_quantum=float(np.std(q, ddof=1)),
        std_classical=float(np.std(c, ddof=1)),
        n_comparisons=n_comparisons,
        significant=p_corrected < 0.05,
    )
