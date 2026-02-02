#!/usr/bin/env python
"""CLI entry point for quantum column generation graph coloring."""

import argparse
import sys

import networkx as nx

from quantum_colgen.column_generation import column_generation, verify_coloring
from quantum_colgen.graphs import TEST_GRAPHS, KNOWN_CHROMATIC, erdos_renyi


def _make_oracle(name: str, **kwargs):
    if name == "classical":
        from quantum_colgen.pricing.classical import ClassicalPricingOracle
        return ClassicalPricingOracle()
    elif name == "pulser":
        from quantum_colgen.pricing.pulser_oracle import PulserPricingOracle
        return PulserPricingOracle(**kwargs)
    elif name == "dirac":
        from quantum_colgen.pricing.dirac_oracle import DiracPricingOracle
        return DiracPricingOracle(**kwargs)
    else:
        raise ValueError(f"Unknown oracle: {name}")


def main():
    parser = argparse.ArgumentParser(
        description="Quantum column generation for minimum vertex graph coloring"
    )
    parser.add_argument(
        "--oracle",
        choices=["classical", "pulser", "dirac"],
        default="classical",
        help="Pricing oracle (default: classical)",
    )
    parser.add_argument("--test", action="store_true", help="Run on predefined test graphs")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--nodes", type=int, default=9, help="Erdos-Renyi node count")
    parser.add_argument("--edge-prob", type=float, default=0.4, help="Erdos-Renyi edge probability")
    parser.add_argument("--num-samples", type=int, default=20, help="Dirac num_samples")
    parser.add_argument("--relax-schedule", type=int, default=2, choices=[1, 2, 3, 4])
    parser.add_argument("--method", default="gibbons", choices=["gibbons", "filter"],
                        help="Dirac method (default: gibbons)")

    args = parser.parse_args()

    oracle_kwargs = {}
    if args.oracle == "dirac":
        oracle_kwargs = {
            "method": args.method,
            "num_samples": args.num_samples,
            "relax_schedule": args.relax_schedule,
        }

    if args.test:
        print("=" * 60)
        print("Column generation on predefined test graphs")
        print("=" * 60)
        all_pass = True
        for name, factory in TEST_GRAPHS.items():
            G = factory()
            expected = KNOWN_CHROMATIC[name]
            oracle = _make_oracle(args.oracle, **oracle_kwargs)
            num_colors, coloring, stats = column_generation(
                G, oracle, verbose=args.verbose
            )
            valid = verify_coloring(G, coloring) if coloring else False
            status = "PASS" if (num_colors == expected and valid) else "FAIL"
            if status == "FAIL":
                all_pass = False
            print(
                f"  {name:<16} chi={num_colors}  expected={expected}  "
                f"valid={valid}  cols={stats.get('columns_generated', '?')}  [{status}]"
            )
        print("=" * 60)
        return 0 if all_pass else 1
    else:
        G = erdos_renyi(args.nodes, args.edge_prob)
        print(f"Erdos-Renyi G({args.nodes}, {args.edge_prob}): "
              f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        oracle = _make_oracle(args.oracle, **oracle_kwargs)
        num_colors, coloring, stats = column_generation(
            G, oracle, verbose=args.verbose
        )
        valid = verify_coloring(G, coloring) if coloring else False
        print(f"Result: {num_colors} colors, valid={valid}")
        for i, c in enumerate(coloring):
            print(f"  Color {i + 1}: {sorted(c)}")
        return 0 if valid else 1


if __name__ == "__main__":
    sys.exit(main())
