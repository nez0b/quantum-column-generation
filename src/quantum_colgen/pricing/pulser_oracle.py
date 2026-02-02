"""Pulser neutral-atom simulator pricing oracle for MWIS."""

from typing import List, Set

import networkx as nx
import numpy as np

from .base import PricingOracle

try:
    from mis import MISInstance, MISSolver, BackendConfig, BackendType
    from mis.pipeline.config import SolverConfig
    from mis.pipeline.pulse import BasePulseShaper, Register, Pulse
    from mis.pipeline.embedder import BaseEmbedder
    from mis.shared.types import MethodType
    import pulser
    from pulser import InterpolatedWaveform, Pulse as PulserPulse

    PULSER_AVAILABLE = True
except ImportError:
    PULSER_AVAILABLE = False


def _center_positions(positions):
    """Center node positions around centroid."""
    all_coords = np.stack(list(positions.values()))
    centroid = np.mean(all_coords, axis=0)
    return {node: pos - centroid for node, pos in positions.items()}


if PULSER_AVAILABLE:

    class _Embedder(BaseEmbedder):
        """Atom register embedder with proper coordinate scaling."""

        def embed(self, instance: MISInstance, config: SolverConfig, backend) -> Register:
            device = backend.device()
            assert device is not None

            positions = nx.get_node_attributes(instance.graph, "pos")
            if not positions:
                positions = nx.spring_layout(instance.graph, iterations=100)
            positions = {k: np.array(v) for k, v in positions.items()}

            if len(positions) > 1:
                positions = _center_positions(positions)

            distances = [
                np.linalg.norm(positions[v1] - positions[v2])
                for v1 in instance.graph.nodes()
                for v2 in instance.graph.nodes()
                if v1 != v2
            ]

            multiplier = 1.0
            max_distance_scaled = device.min_atom_distance

            if distances:
                min_distance = np.min(distances)
                max_distance = np.max(distances)
                if min_distance < device.min_atom_distance:
                    multiplier = device.min_atom_distance / min_distance
                    positions = {i: v * multiplier for i, v in positions.items()}
                max_distance_scaled = max_distance * multiplier

            instance.graph.graph["scaling_multiplier"] = multiplier
            instance.graph.graph["max_distance_scaled"] = max_distance_scaled

            return Register(qubits={f"q{node}": pos for node, pos in positions.items()})

    class _PulseShaper(BasePulseShaper):
        """Pulse shaper with blockade-radius-aware Rabi frequency."""

        def generate(self, config: SolverConfig, register: Register, backend, instance) -> Pulse:
            device = backend.device()
            assert device is not None

            duration_us = self.duration_us
            if duration_us is None:
                duration_us = device.max_sequence_duration

            rb = instance.graph.graph.get("max_distance_scaled", 38)
            omega = device.rabi_from_blockade(rb * 0.75)
            Delta = 2 * omega

            amplitude = InterpolatedWaveform(duration_us, [1e-9, omega, 1e-9])
            detuning = InterpolatedWaveform(duration_us, [-Delta, 0, Delta])
            rydberg_pulse = PulserPulse(amplitude, detuning, 0)
            assert isinstance(rydberg_pulse, PulserPulse)
            return rydberg_pulse


class PulserPricingOracle(PricingOracle):
    """Pricing oracle using the Pulser neutral-atom MIS simulator.

    Filters to positive-dual subgraph, runs quantum MIS, and validates
    profitability of returned independent sets.
    """

    def __init__(self, duration_us: int = 4000, runs: int = 500, max_solutions: int = 5):
        if not PULSER_AVAILABLE:
            raise ImportError(
                "Pulser oracle requires 'pulser' and 'maximum-independent-set' packages."
            )
        self.duration_us = duration_us
        self.runs = runs
        self.max_solutions = max_solutions

    def solve(self, graph: nx.Graph, dual_vars: np.ndarray) -> List[Set[int]]:
        positive_nodes = [v for v in graph.nodes() if dual_vars[v] > 1e-5]
        if not positive_nodes:
            return []

        subgraph = graph.subgraph(positive_nodes).copy()
        if not subgraph.nodes():
            return []

        # Edgeless subgraph — all nodes form one IS
        if not subgraph.edges():
            total = sum(dual_vars[v] for v in subgraph.nodes())
            if total > 1 + 1e-5:
                return [set(subgraph.nodes())]
            return []

        instance = MISInstance(subgraph)
        config = SolverConfig(
            method=MethodType.EAGER,
            backend=BackendConfig(backend_type=BackendType.QUTIP),
            embedder=_Embedder(),
            pulse_shaper=_PulseShaper(duration_us=self.duration_us),
            preprocessor=None,
            max_number_of_solutions=self.max_solutions,
            max_iterations=1,
            runs=self.runs,
        )

        try:
            solver = MISSolver(instance, config)
            reports = solver.solve()
        except Exception:
            return []

        if not reports:
            return []

        seen = set()
        profitable: List[Set[int]] = []
        for report in reports:
            if not report.nodes:
                continue
            sig = tuple(sorted(report.nodes))
            if sig in seen:
                continue
            seen.add(sig)

            nodes = set(sig)
            # Validate independence
            if any(graph.has_edge(u, w) for u in nodes for w in nodes if u != w):
                continue

            total = sum(dual_vars[v] for v in nodes)
            if total > 1 + 1e-5:
                profitable.append(nodes)

        return profitable
