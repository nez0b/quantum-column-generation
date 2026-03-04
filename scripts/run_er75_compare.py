#!/usr/bin/env python3
"""Four-way comparison on ER(75, 0.6, seed=21).

Runs Hexaly ILP, Classical CG, Classical B&P, and Quantum B&P,
merges results into a v2 JSON record, saves to the qbp benchmarks dir,
and ingests to the qbp SQLite DB.

Usage:
    # Dry-run (Hexaly + CG only, ~20 min):
    source ~/.zshrc
    uv run python scripts/run_er75_compare.py --skip-bp --skip-qbp

    # Full run (4+ hr for QBP):
    uv run python scripts/run_er75_compare.py
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import networkx as nx

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quantum_colgen.column_generation import column_generation
from quantum_colgen.direct_ilp import solve_coloring_ilp_hexaly
from quantum_colgen.pricing.classical_lp import ClassicalLPPricingOracle

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N = 75
P = 0.6
SEED = 21
INSTANCE = f"ER({N},{P},{SEED})"
GRAPH_ID = f"er_{N}_{P}_s{SEED}"
TIME_LIMIT = 1200.0   # 20 min for Hexaly/CG/BP
QBP_TIME_LIMIT = 14400.0  # 4 hr for QBP

QBP_DIR = Path(__file__).parent.parent / "quantum-branch-price"
BENCHMARKS_DIR = QBP_DIR / "benchmarks"


# ---------------------------------------------------------------------------
# Graph generation
# ---------------------------------------------------------------------------

def build_graph() -> nx.Graph:
    G = nx.gnp_random_graph(N, P, seed=SEED)
    print(f"Graph: {INSTANCE}  |V|={G.number_of_nodes()}  |E|={G.number_of_edges()}")
    # Save graph metadata
    graph_dir = BENCHMARKS_DIR / f"er{N}_{P}_s{SEED}"
    graph_dir.mkdir(parents=True, exist_ok=True)
    graph_meta = {
        "instance": INSTANCE,
        "n": G.number_of_nodes(),
        "m": G.number_of_edges(),
        "generator": "gnp_random_graph",
        "n_param": N,
        "p_param": P,
        "seed": SEED,
    }
    (graph_dir / "graph_meta.json").write_text(json.dumps(graph_meta, indent=2))
    return G


# ---------------------------------------------------------------------------
# Method runners
# ---------------------------------------------------------------------------

def run_hexaly(G: nx.Graph) -> dict:
    print(f"\n{'='*60}")
    print(f"Running Hexaly ILP (time_limit={TIME_LIMIT}s) ...")
    print(f"{'='*60}")
    t0 = time.monotonic()
    try:
        chi, _, solve_time, info = solve_coloring_ilp_hexaly(G, time_limit=TIME_LIMIT)
    except Exception as e:
        wall = time.monotonic() - t0
        print(f"  Hexaly error: {e}")
        return {
            "method": "Hexaly ILP",
            "chi": None,
            "wall_seconds": round(wall, 2),
            "optimal": False,
            "error": str(e),
        }
    wall = time.monotonic() - t0
    optimal = info.get("status") == "optimal"
    print(f"  chi={chi}  wall={wall:.1f}s  status={info.get('status')}")
    return {
        "method": "Hexaly ILP",
        "chi": chi,
        "wall_seconds": round(wall, 2),
        "optimal": optimal,
    }


def run_cg(G: nx.Graph) -> tuple:
    """Returns (result_dict, coloring) where coloring is list-of-lists for warm-start."""
    print(f"\n{'='*60}")
    print(f"Running Classical CG (LP oracle, time_limit={TIME_LIMIT}s) ...")
    print(f"{'='*60}")
    oracle = ClassicalLPPricingOracle()
    t0 = time.monotonic()
    chi, coloring, stats = column_generation(
        G,
        oracle,
        max_iterations=2000,
        verbose=True,
        time_limit=TIME_LIMIT,
    )
    wall = time.monotonic() - t0
    timer = oracle.timer.summary()
    print(f"  chi={chi}  wall={wall:.1f}s  iters={stats.get('iterations')}  "
          f"cols={stats.get('columns_generated')}  "
          f"time_limit_reached={stats.get('time_limit_reached')}")
    print(f"  Oracle timing: {timer}")
    result = {
        "method": "Classical CG",
        "chi": chi,
        "wall_seconds": round(wall, 2),
        "cg_iterations": stats.get("iterations"),
        "columns_generated": stats.get("columns_generated"),
        "time_limit_reached": stats.get("time_limit_reached", False),
        "oracle_timing": timer,
    }
    # Serialize coloring as list-of-lists (JSON-safe)
    coloring_serializable = [sorted(color_class) for color_class in coloring] if coloring else []
    return result, coloring_serializable


def run_bp_subprocess(
    oracle_type: str, time_limit: float, initial_coloring: list | None = None
) -> dict:
    """Run BP or QBP via subprocess in the qbp venv."""
    label = "QBP" if oracle_type == "dirac" else "BP"
    method_key = "qbp" if oracle_type == "dirac" else "bp"
    print(f"\n{'='*60}")
    print(f"Running {label} (oracle={oracle_type}, time_limit={time_limit}s) ...")
    if initial_coloring:
        print(f"  Warm-start: {len(initial_coloring)} colors from CG")
    print(f"{'='*60}")

    with tempfile.NamedTemporaryFile(
        suffix=".json", prefix=f"{method_key}_result_", delete=False
    ) as tmp:
        tmp_path = tmp.name

    # Write warm-start coloring to a temp file if provided
    coloring_tmp_path = None
    if initial_coloring:
        with tempfile.NamedTemporaryFile(
            suffix=".json", prefix="initial_coloring_", delete=False, mode="w"
        ) as ctmp:
            json.dump(initial_coloring, ctmp)
            coloring_tmp_path = ctmp.name

    cmd = [
        "uv", "run", "python", "scripts/run_bp.py",
        "--graph", GRAPH_ID,
        "--oracle", oracle_type,
        "--time-limit", str(time_limit),
        "--json", tmp_path,
        "--verbose",
    ]
    if coloring_tmp_path:
        cmd += ["--initial-coloring", coloring_tmp_path]
    print(f"  Command: {' '.join(cmd)}")
    print(f"  cwd: {QBP_DIR}")

    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(QBP_DIR),
            capture_output=False,   # stream stdout/stderr to terminal
            timeout=time_limit + 300,  # 5-min grace period
        )
    except subprocess.TimeoutExpired:
        wall = time.monotonic() - t0
        print(f"  {label} subprocess timed out after {wall:.1f}s")
        return {
            "method": f"{'Quantum' if oracle_type == 'dirac' else 'Classical'} B&P",
            "chi": None,
            "wall_seconds": round(wall, 2),
            "error": "subprocess timeout",
        }
    except Exception as e:
        wall = time.monotonic() - t0
        print(f"  {label} subprocess error: {e}")
        return {
            "method": f"{'Quantum' if oracle_type == 'dirac' else 'Classical'} B&P",
            "chi": None,
            "wall_seconds": round(wall, 2),
            "error": str(e),
        }

    wall = time.monotonic() - t0

    if proc.returncode != 0:
        print(f"  {label} exited with code {proc.returncode}")
        return {
            "method": f"{'Quantum' if oracle_type == 'dirac' else 'Classical'} B&P",
            "chi": None,
            "wall_seconds": round(wall, 2),
            "error": f"exit code {proc.returncode}",
        }

    # Parse JSON output
    try:
        data = json.loads(Path(tmp_path).read_text())
        method_data = data[0]["methods"][method_key]
        print(f"  chi={method_data.get('chi')}  wall={method_data.get('wall_seconds')}s  "
              f"optimal={method_data.get('optimal')}")
        return method_data
    except Exception as e:
        print(f"  Failed to parse {label} result JSON: {e}")
        return {
            "method": f"{'Quantum' if oracle_type == 'dirac' else 'Classical'} B&P",
            "chi": None,
            "wall_seconds": round(wall, 2),
            "error": f"json parse error: {e}",
        }
    finally:
        Path(tmp_path).unlink(missing_ok=True)
        if coloring_tmp_path:
            Path(coloring_tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ER(75,0.6,21) four-way comparison")
    parser.add_argument("--skip-bp", action="store_true",
                        help="Skip Classical B&P")
    parser.add_argument("--skip-qbp", action="store_true",
                        help="Skip Quantum B&P (Dirac)")
    parser.add_argument("--skip-hexaly", action="store_true",
                        help="Skip Hexaly ILP")
    parser.add_argument("--skip-cg", action="store_true",
                        help="Skip Classical CG")
    args = parser.parse_args()

    G = build_graph()
    n = G.number_of_nodes()
    m = G.number_of_edges()

    methods: dict = {}
    cg_coloring: list | None = None  # warm-start for BP/QBP

    # --- Hexaly ---
    if not args.skip_hexaly:
        methods["hexaly"] = run_hexaly(G)
    else:
        print("Skipping Hexaly.")

    # --- CG ---
    if not args.skip_cg:
        methods["cg"], cg_coloring = run_cg(G)
    else:
        print("Skipping Classical CG.")

    # --- BP ---
    if not args.skip_bp:
        methods["bp"] = run_bp_subprocess("classical", TIME_LIMIT, cg_coloring)
    else:
        print("Skipping Classical B&P.")

    # --- QBP ---
    if not args.skip_qbp:
        methods["qbp"] = run_bp_subprocess("dirac", QBP_TIME_LIMIT, cg_coloring)
    else:
        print("Skipping Quantum B&P.")

    # --- Build v2 JSON record ---
    record = [{
        "instance": INSTANCE,
        "n": n,
        "m": m,
        "methods": methods,
    }]

    # --- Save JSON ---
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_filename = f"er75_compare_{ts}.json"
    out_path = BENCHMARKS_DIR / out_filename
    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2))
    print(f"\n{'='*60}")
    print(f"Results saved to: {out_path}")

    # --- Ingest to DB ---
    print("\nIngesting to DB ...")
    ingest_cmd = [
        "uv", "run", "python", "scripts/benchmark_db.py",
        "ingest", str(out_path),
    ]
    try:
        subprocess.run(ingest_cmd, cwd=str(QBP_DIR), check=True)
    except subprocess.CalledProcessError as e:
        print(f"  DB ingest failed: {e}")

    # --- Print comparison table ---
    print(f"\n{'='*60}")
    print(f"Comparison: {INSTANCE}")
    print(f"{'Method':<20} {'chi':>6} {'wall_s':>10} {'optimal':>9}")
    print("-" * 50)
    for key, data in methods.items():
        chi = data.get("chi", "N/A")
        wall = data.get("wall_seconds", "N/A")
        opt = data.get("optimal", "?")
        label = data.get("method", key)
        if isinstance(wall, float):
            wall_str = f"{wall:.1f}"
        else:
            wall_str = str(wall)
        print(f"{label:<20} {str(chi):>6} {wall_str:>10} {str(opt):>9}")

    # Verify QBP device_seconds if present
    if "qbp" in methods:
        qbp_timing = methods["qbp"].get("oracle_timing", {})
        device_s = qbp_timing.get("total_device_seconds", 0)
        print(f"\nQBP device_seconds: {device_s}")
        if device_s == 0:
            print("  WARNING: total_device_seconds=0 — Dirac device timing may not be captured.")
        else:
            print("  OK: Dirac device timing is non-zero.")


if __name__ == "__main__":
    main()
