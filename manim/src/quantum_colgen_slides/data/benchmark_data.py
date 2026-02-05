"""Benchmark data for the presentation.

Data sourced from results/lp_vs_dirac_large.json and related benchmarks.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class BenchmarkResult:
    """Single benchmark result for a method on a graph."""

    method: str  # "greedy", "milp", "lp", "dirac"
    graph_name: str
    n_nodes: int
    n_edges: int
    chi: int  # chromatic number found
    iterations: Optional[int] = None
    cols_per_call: Optional[float] = None
    wall_seconds: Optional[float] = None


# ER(40, 0.3) results - Dirac wins by 2 colors!
ER_40_03_RESULTS = [
    BenchmarkResult("greedy", "ER(40,0.3)", 40, 244, chi=8),
    BenchmarkResult(
        "milp", "ER(40,0.3)", 40, 244, chi=8, iterations=130, cols_per_call=1.0
    ),
    BenchmarkResult(
        "lp", "ER(40,0.3)", 40, 244, chi=8, iterations=19, cols_per_call=10.25
    ),
    BenchmarkResult(
        "dirac", "ER(40,0.3)", 40, 244, chi=6, iterations=13, cols_per_call=20.79
    ),
]

# ER(40, 0.5) results
ER_40_05_RESULTS = [
    BenchmarkResult("greedy", "ER(40,0.5)", 40, 385, chi=11),
    BenchmarkResult(
        "milp", "ER(40,0.5)", 40, 385, chi=10, iterations=101, cols_per_call=1.0
    ),
    BenchmarkResult(
        "lp", "ER(40,0.5)", 40, 385, chi=9, iterations=12, cols_per_call=10.54
    ),
    BenchmarkResult(
        "dirac", "ER(40,0.5)", 40, 385, chi=9, iterations=16, cols_per_call=12.12
    ),
]

# ER(50, 0.3) results - Dirac matches greedy
ER_50_03_RESULTS = [
    BenchmarkResult("greedy", "ER(50,0.3)", 50, 368, chi=8),
    BenchmarkResult(
        "milp", "ER(50,0.3)", 50, 368, chi=11, iterations=189, cols_per_call=1.0
    ),
    BenchmarkResult(
        "lp", "ER(50,0.3)", 50, 368, chi=10, iterations=39, cols_per_call=8.18
    ),
    BenchmarkResult(
        "dirac", "ER(50,0.3)", 50, 368, chi=8, iterations=18, cols_per_call=29.74
    ),
]

# ER(50, 0.5) results
ER_50_05_RESULTS = [
    BenchmarkResult("greedy", "ER(50,0.5)", 50, 596, chi=11),
    BenchmarkResult(
        "milp", "ER(50,0.5)", 50, 596, chi=11, iterations=137, cols_per_call=1.0
    ),
    BenchmarkResult(
        "lp", "ER(50,0.5)", 50, 596, chi=10, iterations=23, cols_per_call=8.67
    ),
    BenchmarkResult(
        "dirac", "ER(50,0.5)", 50, 596, chi=10, iterations=14, cols_per_call=22.67
    ),
]

# All results grouped by graph
ALL_BENCHMARK_RESULTS = {
    "ER(40,0.3)": ER_40_03_RESULTS,
    "ER(40,0.5)": ER_40_05_RESULTS,
    "ER(50,0.3)": ER_50_03_RESULTS,
    "ER(50,0.5)": ER_50_05_RESULTS,
}

# Hero slide data (ER(40,0.3) where Dirac wins by 2 colors)
HERO_GRAPH = "ER(40,0.3)"
HERO_RESULTS = ER_40_03_RESULTS

# Key statistics for summary
DIRAC_STATS = {
    "wins": 4,  # Dirac strictly better chi
    "ties": 4,  # Dirac matches best
    "losses": 0,  # Dirac worse than best
    "avg_cols_per_call": 21.3,  # Average across all runs
    "iteration_reduction": "10x",  # Compared to MILP
}

# Multi-threshold extraction thresholds used by Dirac
EXTRACTION_THRESHOLDS = [0.005, 0.01, 0.05, 0.1, 0.2]


def get_method_display_name(method: str) -> str:
    """Convert method code to display name."""
    names = {
        "greedy": "Greedy",
        "milp": "MILP",
        "lp": "LP Relaxation",
        "dirac": "Dirac-3",
    }
    return names.get(method, method)


def get_best_chi(results: List[BenchmarkResult]) -> int:
    """Get the best (lowest) chi from a list of results."""
    return min(r.chi for r in results)


def is_winner(result: BenchmarkResult, results: List[BenchmarkResult]) -> bool:
    """Check if this result has the best chi in the list."""
    best = get_best_chi(results)
    return result.chi == best
