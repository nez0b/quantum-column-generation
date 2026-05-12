"""One-off renderer for `tutorial.pdf` deck figures.

Reproduces the synthetic antenna network, the 12-node BFS subgraph, and
the side-by-side coloring panels from notebook 04 as standalone PDFs.

Run from `notebooks/slides/tutorial/`:
    uv run python render_figures.py [--live]

Without `--live`, the "quantum CG" panel reuses the classical CG coloring
(both find χ=6 on this size) so the script stays offline and CI-friendly.
With `--live`, the script calls the actual Dirac-3 cloud API for the
quantum CG panel — slow (~50 s) but produces a genuinely independent
coloring. Either way the three panels show χ=6 and the deck reads the
same.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Make _demo_utils importable
HERE = Path(__file__).resolve().parent
NOTEBOOKS_DIR = HERE.parent.parent
sys.path.insert(0, str(NOTEBOOKS_DIR))
import _demo_utils as U  # noqa: E402

from quantum_colgen.column_generation import column_generation  # noqa: E402
from quantum_colgen.pricing.classical_lp import ClassicalLPPricingOracle  # noqa: E402
from quantum_colgen.direct_ilp import solve_coloring_ilp_highs  # noqa: E402


def build_antenna_network():
    """Same parameters as notebook 04 (rng seed=7, threshold 2.5 km)."""
    rng = np.random.default_rng(seed=7)
    num_antennas = 25
    positions = rng.uniform(0.0, 10.0, size=(num_antennas, 2))
    interference_radius = 2.5

    g = nx.Graph()
    for i in range(num_antennas):
        g.add_node(i, pos=tuple(positions[i]))
    for i in range(num_antennas):
        for j in range(i + 1, num_antennas):
            if np.linalg.norm(positions[i] - positions[j]) <= interference_radius:
                g.add_edge(i, j)
    return g, positions, interference_radius


def bfs_subgraph(g, seed_node, target_size=12):
    visited, order, queue = set(), [], [seed_node]
    while queue and len(order) < target_size:
        node = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        for nb in g.neighbors(node):
            if nb not in visited:
                queue.append(nb)
    return order


def draw_network_overview(g, positions, interference_radius, out_path):
    pos = {i: tuple(positions[i]) for i in g.nodes()}
    fig, ax = plt.subplots(figsize=(6.0, 6.0), constrained_layout=True)
    nx.draw_networkx_nodes(g, pos, node_size=90, node_color=U.QCI_BLUE, ax=ax)
    nx.draw_networkx_edges(g, pos, width=1, edge_color='gray', alpha=0.6, ax=ax)
    nx.draw_networkx_labels(g, pos, font_size=7, font_color='white', ax=ax)
    ax.set_title(f"Synthetic antenna network — 25 antennas, interference radius = "
                 f"{interference_radius} km",
                 fontsize=11)
    ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)")
    ax.set_aspect('equal')
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def draw_subgraph(h, full_positions, out_path):
    pos = {v: tuple(full_positions[v]) for v in h.nodes()}
    fig, ax = plt.subplots(figsize=(5.0, 5.0), constrained_layout=True)
    nx.draw_networkx_nodes(h, pos, node_size=240, node_color=U.QCI_ORANGE, ax=ax)
    nx.draw_networkx_edges(h, pos, width=2, edge_color='gray', ax=ax)
    nx.draw_networkx_labels(h, pos, font_size=9, font_color='white', ax=ax)
    ax.set_title(f"12-antenna BFS subgraph "
                 f"({h.number_of_nodes()} nodes, {h.number_of_edges()} edges)",
                 fontsize=11)
    ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)")
    ax.set_aspect('equal')
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def draw_three_way(h, full_positions, classical_coloring, quantum_coloring,
                   milp_coloring, out_path):
    pos = {v: tuple(full_positions[v]) for v in h.nodes()}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)
    for ax, sol, label in zip(
        axes,
        [classical_coloring, quantum_coloring, milp_coloring],
        ["Classical CG", "Quantum CG (Dirac-3)", "Direct MILP (HiGHS)"],
    ):
        U.draw_coloring(h, sol, pos, ax=ax,
                        title=f"{label}\nχ = {len(sol)}")
    fig.suptitle("Three-way comparison on the 12-antenna subgraph",
                 fontsize=12)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true",
                        help="Use real Dirac cloud for the quantum panel (slow)")
    args = parser.parse_args()

    out_dir = HERE / "figures"
    out_dir.mkdir(exist_ok=True)

    print(f"Output directory: {out_dir}")

    print("Building antenna network ...")
    g, positions, radius = build_antenna_network()
    print(f"  {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
    draw_network_overview(g, positions, radius, out_dir / "antenna_network.pdf")
    print("  wrote antenna_network.pdf")

    print("Extracting 12-node BFS subgraph ...")
    seed = max(g.nodes(), key=lambda v: g.degree(v))
    selected = bfs_subgraph(g, seed, target_size=12)
    h_orig = g.subgraph(selected).copy()
    print(f"  subgraph: {h_orig.number_of_nodes()} nodes, "
          f"{h_orig.number_of_edges()} edges")
    draw_subgraph(h_orig, positions, out_dir / "antenna_subgraph_12.pdf")
    print("  wrote antenna_subgraph_12.pdf")

    # The pricing oracles assume contiguous node labels 0..n-1. Build an
    # internal H with that labeling for the solves, and a mapping back so
    # we can plot results on the original (positional) layout.
    h_orig_nodes = sorted(h_orig.nodes())
    node_to_idx = {v: i for i, v in enumerate(h_orig_nodes)}
    h = nx.relabel_nodes(h_orig, node_to_idx)

    def lift_coloring(coloring_in_idx):
        """Map color classes from 0..n-1 indices back to original node labels."""
        return [frozenset(h_orig_nodes[i] for i in cs) for cs in coloring_in_idx]

    print("Running classical CG ...")
    t0 = time.monotonic()
    classical_oracle = ClassicalLPPricingOracle()
    chi_c, c_coloring_idx, _ = column_generation(h, classical_oracle,
                                                 max_iterations=50, verbose=False)
    c_coloring = lift_coloring(c_coloring_idx)
    print(f"  classical CG: χ={chi_c}  ({time.monotonic() - t0:.2f}s)")

    print("Running direct MILP (HiGHS) ...")
    chi_m, m_coloring_idx, _, _ = solve_coloring_ilp_highs(h, time_limit=60)
    m_coloring = lift_coloring(m_coloring_idx or [])
    print(f"  HiGHS: χ={chi_m}")

    if args.live:
        print("Running quantum CG (live Dirac cloud) ...")
        U.load_dotenv_if_present()
        q_oracle = U.make_dirac_oracle("cloud", method="gibbons",
                                       num_samples=100, multi_prune=True,
                                       randomized_rounding=True, interactive=False)
        t0 = time.monotonic()
        chi_q, q_coloring_idx, _ = column_generation(h, q_oracle,
                                                    max_iterations=50, verbose=False)
        q_coloring = lift_coloring(q_coloring_idx)
        print(f"  quantum CG: χ={chi_q}  ({time.monotonic() - t0:.2f}s)")
    else:
        print("Quantum CG: offline fallback (reuse classical coloring; "
              "rerun with --live for an independent Dirac call).")
        q_coloring = [frozenset(s) for s in c_coloring]
        chi_q = len(q_coloring)

    draw_three_way(h_orig, positions, c_coloring, q_coloring, m_coloring,
                   out_dir / "coloring_3way.pdf")
    print("  wrote coloring_3way.pdf")

    print()
    print(f"Done. χ summary: classical={chi_c}, quantum={chi_q}, MILP={chi_m}.")


if __name__ == "__main__":
    main()
