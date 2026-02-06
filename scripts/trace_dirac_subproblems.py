"""Trace Dirac CG iteration-by-iteration: subproblem sizes, columns, timing."""

import json
import time
import sys
import argparse

import networkx as nx
import numpy as np

from quantum_colgen.master_problem import solve_rmp, solve_final_ilp
from quantum_colgen.pricing.dirac_oracle import DiracPricingOracle


def run_traced_cg(graph, oracle, max_iterations=500, verbose=True):
    """Column generation with per-iteration tracing of subproblem sizes."""
    node_list = sorted(graph.nodes())
    num_vertices = len(node_list)
    num_edges = graph.number_of_edges()

    columns = [frozenset([i]) for i in range(num_vertices)]
    known_sigs = {tuple(sorted(c)) for c in columns}

    iteration_log = []

    print(f"\n{'='*80}")
    print(f"Graph: n={num_vertices}, m={num_edges}")
    print(f"{'='*80}")
    print(f"{'Iter':>4} | {'RMP obj':>8} | {'Pos-dual':>8} | {'Subgraph':>10} | "
          f"{'Sub edges':>9} | {'Cols added':>10} | {'Total cols':>10} | {'API time':>8}")
    print(f"{'-'*4:>4}-+-{'-'*8:>8}-+-{'-'*8:>8}-+-{'-'*10:>10}-+-"
          f"{'-'*9:>9}-+-{'-'*10:>10}-+-{'-'*10:>10}-+-{'-'*8:>8}")

    for iteration in range(1, max_iterations + 1):
        # Solve RMP
        obj, dual_vars = solve_rmp(columns, num_vertices)
        if dual_vars is None:
            print(f"  RMP failed at iteration {iteration}")
            break

        # Analyze the subproblem BEFORE calling oracle
        pos_nodes = [v for v in graph.nodes() if dual_vars[v] > 1e-5]
        subgraph = graph.subgraph(pos_nodes)
        sub_n = len(pos_nodes)
        sub_m = subgraph.number_of_edges()

        # Dual stats
        duals_pos = dual_vars[dual_vars > 1e-5]
        dual_min = duals_pos.min() if len(duals_pos) > 0 else 0
        dual_max = duals_pos.max() if len(duals_pos) > 0 else 0
        dual_mean = duals_pos.mean() if len(duals_pos) > 0 else 0

        # Call oracle
        oracle.timer.reset()
        t0 = time.monotonic()
        new_cols = oracle.solve(graph, dual_vars)
        oracle_time = time.monotonic() - t0

        if not new_cols:
            print(f"{iteration:4d} | {obj:8.4f} | {sub_n:8d} | {sub_n:10d} | "
                  f"{sub_m:9d} | {'converged':>10} | {len(columns):10d} | {oracle_time:7.1f}s")

            iter_data = {
                "iteration": iteration,
                "rmp_obj": round(obj, 4),
                "pos_dual_nodes": sub_n,
                "subgraph_nodes": sub_n,
                "subgraph_edges": sub_m,
                "cols_added": 0,
                "total_cols": len(columns),
                "oracle_time_s": round(oracle_time, 2),
                "dual_min": round(dual_min, 4),
                "dual_max": round(dual_max, 4),
                "dual_mean": round(dual_mean, 4),
                "converged": True,
            }
            iteration_log.append(iter_data)
            break

        added = 0
        col_sizes = []
        for col_set in new_cols:
            sig = tuple(sorted(col_set))
            if sig not in known_sigs:
                columns.append(frozenset(col_set))
                known_sigs.add(sig)
                added += 1
                col_sizes.append(len(col_set))

        api_time = oracle.timer.total_api_seconds if hasattr(oracle.timer, 'total_api_seconds') else oracle_time

        print(f"{iteration:4d} | {obj:8.4f} | {sub_n:8d} | {sub_n:10d} | "
              f"{sub_m:9d} | {added:10d} | {len(columns):10d} | {oracle_time:7.1f}s")

        iter_data = {
            "iteration": iteration,
            "rmp_obj": round(obj, 4),
            "pos_dual_nodes": sub_n,
            "subgraph_nodes": sub_n,
            "subgraph_edges": sub_m,
            "cols_added": added,
            "total_cols": len(columns),
            "oracle_time_s": round(oracle_time, 2),
            "dual_min": round(dual_min, 4),
            "dual_max": round(dual_max, 4),
            "dual_mean": round(dual_mean, 4),
            "col_sizes": col_sizes,
            "converged": False,
        }
        iteration_log.append(iter_data)

        if added == 0:
            print(f"  All columns already known — converged")
            iter_data["converged"] = True
            break

    # Final ILP
    print(f"\n{'='*80}")
    print(f"Solving final ILP with {len(columns)} columns...")
    num_colors, selected_indices = solve_final_ilp(columns, num_vertices)

    if num_colors is not None:
        coloring = [columns[i] for i in selected_indices]
        print(f"Result: χ = {num_colors}")
        print(f"Color classes: {[sorted(c) for c in coloring]}")
    else:
        print("Final ILP failed")

    # Summary
    print(f"\n{'='*80}")
    print("SUBPROBLEM SIZE TRACE")
    print(f"{'='*80}")
    print(f"Original graph: n={num_vertices}, m={num_edges}")
    print(f"Iterations: {len(iteration_log)}")
    if iteration_log:
        sizes = [d["subgraph_nodes"] for d in iteration_log]
        edges = [d["subgraph_edges"] for d in iteration_log]
        print(f"Subproblem nodes: {sizes}")
        print(f"Subproblem edges: {edges}")
        print(f"Node range: {min(sizes)} — {max(sizes)}  (full graph = {num_vertices})")
        print(f"Edge range: {min(edges)} — {max(edges)}  (full graph = {num_edges})")

    return iteration_log, num_colors


def main():
    parser = argparse.ArgumentParser(description="Trace Dirac CG subproblem sizes")
    parser.add_argument("--graph", type=str, default="test",
                        help="Graph to use: 'test', 'erN_P' (e.g. er40_0.3), or benchmark dir")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--save", type=str, default=None, help="Save trace to JSON file")
    args = parser.parse_args()

    # Build graph
    if args.graph == "test":
        from quantum_colgen.graphs import paper_5vertex
        graph = paper_5vertex()
    elif args.graph.startswith("er"):
        # Try loading from benchmarks
        bench_path = f"benchmarks/{args.graph}/graph.json"
        try:
            with open(bench_path) as f:
                data = json.load(f)
            if "nodes" in data:
                graph = nx.node_link_graph(data)
            else:
                # Custom format: {n_nodes, edges: [[u,v], ...]}
                graph = nx.Graph()
                graph.add_nodes_from(range(data["n_nodes"]))
                graph.add_edges_from(data["edges"])
            print(f"Loaded graph from {bench_path}")
        except FileNotFoundError:
            # Generate fresh
            parts = args.graph.replace("er", "").split("_")
            n, p = int(parts[0]), float(parts[1])
            graph = nx.erdos_renyi_graph(n, p, seed=42)
            print(f"Generated ER({n},{p}) with seed=42")
    else:
        with open(args.graph) as f:
            data = json.load(f)
        graph = nx.node_link_graph(data)

    # Relabel to 0..n-1
    graph = nx.convert_node_labels_to_integers(graph)

    oracle = DiracPricingOracle(
        method="gibbons",
        num_samples=args.num_samples,
        multi_prune=True,
        randomized_rounding=True,
        num_random_rounds=10,
        random_seed=42,
    )

    trace, chi = run_traced_cg(graph, oracle, max_iterations=args.max_iterations)

    if args.save:
        with open(args.save, "w") as f:
            json.dump({"graph": args.graph, "chi": chi, "trace": trace}, f, indent=2)
        print(f"\nTrace saved to {args.save}")


if __name__ == "__main__":
    main()
