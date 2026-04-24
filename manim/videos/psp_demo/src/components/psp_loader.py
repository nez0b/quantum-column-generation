"""Load psp_*.json files produced by slides/qcg_vs_cg_demo/pipeline/build_psp_data.py."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Repo root: manim/src/quantum_colgen_slides/components/psp_loader.py → parents[4]
REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = REPO_ROOT / "RF-branching" / "slides" / "qcg_vs_cg_demo" / "data"
PSP1_PATH = DATA_DIR / "psp_01.json"
PSP2_PATH = DATA_DIR / "psp_02.json"


@dataclass
class PSPData:
    instance_id: str
    n: int
    m: int
    graph_edges: List[Tuple[int, int]]
    node_list: List[int]
    layout: Dict[int, Tuple[float, float]]
    psp_label: str
    dirac_method: str
    active_nodes: List[int]
    duals: Dict[int, float]
    dirac_columns: List[List[int]]
    dirac_stats: Dict[str, Any]
    lp_columns: List[List[int]]
    lp_stats: Dict[str, Any]

    @property
    def max_dual(self) -> float:
        return max(self.duals.values()) if self.duals else 1.0


def load_psp(path: Path) -> PSPData:
    raw = json.loads(Path(path).read_text())
    layout = {int(k): tuple(v) for k, v in raw["layout"].items()}
    duals = {int(k): float(v) for k, v in raw["subproblem"]["dual_by_label"].items()}
    return PSPData(
        instance_id=raw["instance_id"],
        n=int(raw["n"]),
        m=int(raw["m"]),
        graph_edges=[tuple(e) for e in raw["graph_edges"]],
        node_list=list(raw["node_list"]),
        layout=layout,
        psp_label=raw["psp_label"],
        dirac_method=raw.get("dirac_method", "qcg"),
        active_nodes=list(raw["subproblem"]["internal_node_list"]),
        duals=duals,
        dirac_columns=[list(c) for c in raw["dirac"]["columns"]],
        dirac_stats=raw["dirac"]["columns_stats"],
        lp_columns=[list(c) for c in raw["classical"]["columns"]],
        lp_stats=raw["classical"]["columns_stats"],
    )
