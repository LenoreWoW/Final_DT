#!/usr/bin/env python3
"""
Run all quantum vs classical benchmarks and save results.

Executes run_benchmark() for each of the 6 healthcare modules with 30 trials
(seed=42) and persists raw per-run CSVs and a JSON summary to benchmark_results/.

Usage:
    python scripts/run_benchmarks.py              # run all modules
    python scripts/run_benchmarks.py module1 ...   # run specific modules only
"""

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalars and arrays."""

    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Ensure project root is on sys.path so backend imports work.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.engine.benchmark_runner import run_benchmark  # noqa: E402

ALL_MODULES = [
    "personalized_medicine",
    "drug_discovery",
    "medical_imaging",
    "genomic_analysis",
    "epidemic_modeling",
    "hospital_operations",
]

# Modules with expensive classical baselines (pure-Python CNN / large simulations)
# get fewer runs to keep total wall-clock under ~10 minutes.
_SLOW_MODULES = {"medical_imaging", "drug_discovery"}
N_RUNS_DEFAULT = 30
N_RUNS_SLOW = 10
SEED = 42
OUTPUT_DIR = PROJECT_ROOT / "benchmark_results"


def log(msg: str = ""):
    """Print with immediate flush so progress is visible in non-tty contexts."""
    print(msg, flush=True)


def save_raw_csv(report, outdir: Path) -> Path:
    """Write per-run data to benchmark_results/raw_{module}.csv."""
    path = outdir / f"raw_{report.module_id}.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run",
            "quantum_time",
            "classical_time",
            "quantum_accuracy",
            "classical_accuracy",
        ])
        for i in range(report.n_runs):
            writer.writerow([
                i + 1,
                report.quantum_times[i],
                report.classical_times[i],
                report.quantum_accuracies[i],
                report.classical_accuracies[i],
            ])
    return path


def build_summary_entry(report) -> dict:
    """Convert a BenchmarkReport to a JSON-serializable dict for summary.json."""
    entry = report.to_dict()
    # Add convenience fields that match the router's BENCHMARK_RESULTS schema
    entry["classical_time_seconds"] = float(np.mean(report.classical_times))
    entry["quantum_time_seconds"] = float(np.mean(report.quantum_times))
    entry["classical_accuracy"] = float(np.mean(report.classical_accuracies))
    entry["quantum_accuracy"] = float(np.mean(report.quantum_accuracies))
    entry["speedup"] = report.mean_speedup
    entry["improvement"] = report.mean_accuracy_improvement
    entry["source"] = "experimental"
    return entry


def main():
    # Allow running specific modules from CLI args
    if len(sys.argv) > 1:
        modules = [m for m in sys.argv[1:] if m in ALL_MODULES]
        if not modules:
            log(f"Unknown modules: {sys.argv[1:]}. Valid: {ALL_MODULES}")
            sys.exit(1)
    else:
        modules = ALL_MODULES

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing summary if running incrementally
    summary_path = OUTPUT_DIR / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
    else:
        summary = {}

    log(f"Running benchmarks for {len(modules)} modules (seed={SEED})")
    log(f"Output directory: {OUTPUT_DIR}")
    log()

    overall_start = time.time()

    for module_id in modules:
        n_runs = N_RUNS_SLOW if module_id in _SLOW_MODULES else N_RUNS_DEFAULT
        log(f"--- {module_id} ({n_runs} runs) ---")
        t0 = time.time()
        report = run_benchmark(module_id, n_runs=n_runs, seed=SEED)
        elapsed = time.time() - t0

        csv_path = save_raw_csv(report, OUTPUT_DIR)
        summary[module_id] = build_summary_entry(report)

        log(f"  Completed in {elapsed:.1f}s")
        log(f"  Mean speedup:              {report.mean_speedup:.4f}x")
        log(f"  Mean accuracy improvement: {report.mean_accuracy_improvement:+.4f}")
        if report.statistical_result:
            sr = report.statistical_result
            log(f"  p-value (corrected):       {sr.p_value_corrected:.6f}")
            log(f"  Cohen's d:                 {sr.cohens_d:.4f}")
            log(f"  Significant:               {sr.significant}")
        log(f"  Raw CSV saved to: {csv_path}")
        log()

        # Save summary after each module so partial results survive interruption
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, cls=_NumpyEncoder)

    total_elapsed = time.time() - overall_start
    log(f"All benchmarks complete in {total_elapsed:.1f}s")
    log(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
