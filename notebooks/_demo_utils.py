"""Shared utilities for the quantum-column-generation notebook series.

Three groups of helpers:

1. **Data loaders / visualization** — read the bundled PSP JSONs from the
   slides demo and produce the "duals heatmap + columns grid + coloring"
   figures used across the notebooks.

2. **Backend selection** — a single ``make_dirac_oracle(mode=...)`` factory
   that dispatches to replay (read pickled raw samples), cloud (QCI cloud
   API via ``QCI_TOKEN``), or direct (eqc-direct on-prem) modes. Cloud and
   direct prompt for credentials interactively if env vars are missing.

3. **Run capture / replay round-trip** — ``CapturingPricingOracle`` wraps any
   live pricing oracle, saving each Dirac call to ``notebooks/runs/<ts>/``
   in the same pickle schema used by ``RF-branching/instances/``. The
   companion ``replay_oracle_from_run()`` rebuilds a replay oracle from
   that directory so users can re-run their CG runs offline.

Notebooks should ``from _demo_utils import *`` (or import named symbols).
"""

from __future__ import annotations

import getpass
import json
import os
import pickle
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from unittest import mock

import networkx as nx
import numpy as np

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
NOTEBOOK_DATA_DIR = NOTEBOOKS_DIR / "data"
NOTEBOOK_RUNS_DIR = NOTEBOOKS_DIR / "runs"
SLIDES_DEMO_DIR = REPO_ROOT / "RF-branching" / "slides" / "qcg_vs_cg_demo"

# Prefer the vendored data inside notebooks/ so the bundle ships standalone.
# Fall back to the parent repo's RF-branching layout for developers working
# inside the full repo without data_bundle/.
_DATA_BUNDLE = NOTEBOOKS_DIR / "data_bundle"
DEFAULT_BUNDLED_INSTANCE = "er_n20_p70_s0"
DEFAULT_BUNDLED_METHOD = "qcg"

SLIDES_DATA_DIR = (
    _DATA_BUNDLE if (_DATA_BUNDLE / "psp_01.json").exists()
    else SLIDES_DEMO_DIR / "data"
)
RAW_SAMPLES_ROOT = (
    _DATA_BUNDLE if (_DATA_BUNDLE / DEFAULT_BUNDLED_INSTANCE).is_dir()
    else REPO_ROOT / "RF-branching" / "instances"
)

# Make src/ importable for when the notebooks are run with the project venv.
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

# RF-branching/src on sys.path so pickle.load can resolve the
# `rf_branching.sample_io.CallRecord` class referenced inside bundled call_*.pkl files.
# rf_branching/__init__.py is empty, and sample_io itself depends only on numpy/pickle —
# no qbp / networkx>=3 deps are pulled in by this side-effect.
_RF_SRC = REPO_ROOT / "RF-branching" / "src"
if _RF_SRC.exists() and str(_RF_SRC) not in sys.path:
    sys.path.insert(0, str(_RF_SRC))


# ---------------------------------------------------------------------------
# Color palette (matches the slide deck for visual consistency)
# ---------------------------------------------------------------------------

QCI_BLUE = "#004682"
QCI_TEAL = "#008080"
QCI_ORANGE = "#DC7814"
QCI_GREEN = "#288C3C"
GREY = "#B8B8B8"
DARK_GREY = "#606060"
IS_PALETTE = [
    QCI_BLUE, QCI_TEAL, QCI_ORANGE, QCI_GREEN, "#8E44AD",
    "#D35400", "#117A65", "#884EA0", "#1F618D", "#CA6F1E",
    "#7D6608", "#512E5F", "#0E6251",
]


# ---------------------------------------------------------------------------
# PSP JSON loaders
# ---------------------------------------------------------------------------

def load_psp(psp_id: int, data_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Load a single PSP JSON from the bundled slides data directory."""
    data_dir = Path(data_dir) if data_dir else SLIDES_DATA_DIR
    path = data_dir / f"psp_{psp_id:02d}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"PSP file not found: {path}. Bundled demo data lives in "
            f"{SLIDES_DATA_DIR} — make sure the RF-branching submodule is checked out."
        )
    return json.loads(path.read_text())


def psp_to_graph(psp: Dict[str, Any]) -> nx.Graph:
    g = nx.Graph()
    for v in psp["node_list"]:
        g.add_node(int(v))
    for u, v in psp["graph_edges"]:
        g.add_edge(int(u), int(v))
    return g


def psp_to_layout(psp: Dict[str, Any]) -> Dict[int, Tuple[float, float]]:
    return {int(v): (float(p[0]), float(p[1])) for v, p in psp["layout"].items()}


def psp_dual_array(psp: Dict[str, Any], num_nodes: Optional[int] = None) -> np.ndarray:
    """Convert ``dual_by_label`` (str-keyed) into a dense numpy array."""
    duals_by_label = psp["subproblem"]["dual_by_label"]
    n = num_nodes or psp["n"]
    arr = np.zeros(n, dtype=np.float64)
    for k, v in duals_by_label.items():
        arr[int(k)] = float(v)
    return arr


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _clean_ax(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _mix(a: str, b: str, t: float) -> Tuple[float, float, float]:
    import matplotlib.colors as mc

    ca = np.array(mc.to_rgb(a))
    cb = np.array(mc.to_rgb(b))
    return tuple((1 - t) * ca + t * cb)


def draw_graph(
    ax,
    graph: nx.Graph,
    layout: Dict[int, Tuple[float, float]],
    *,
    node_color: Any = None,
    node_size: Any = 320,
    highlight_nodes: Optional[Iterable[int]] = None,
    highlight_color: str = QCI_ORANGE,
    title: Optional[str] = None,
    edge_color: str = "#888",
    edge_alpha: float = 0.5,
    label_fontsize: int = 7,
) -> None:
    """Draw a single graph on a matplotlib Axes."""
    for u, v in graph.edges():
        x = [layout[u][0], layout[v][0]]
        y = [layout[u][1], layout[v][1]]
        ax.plot(x, y, color=edge_color, alpha=edge_alpha, linewidth=0.9, zorder=1)

    xs, ys = zip(*[layout[v] for v in graph.nodes()])
    if isinstance(node_color, list):
        ax.scatter(xs, ys, s=node_size, c=node_color, edgecolors="black",
                   linewidths=0.8, zorder=2)
    else:
        ax.scatter(xs, ys, s=node_size, c=node_color or "white",
                   edgecolors="black", linewidths=0.8, zorder=2)

    for v, (x, y) in layout.items():
        ax.text(x, y, str(v), ha="center", va="center", fontsize=label_fontsize,
                zorder=3, color="black")

    if highlight_nodes:
        hn = list(highlight_nodes)
        hxs = [layout[v][0] for v in hn]
        hys = [layout[v][1] for v in hn]
        ns = node_size if not isinstance(node_size, list) else max(node_size)
        ax.scatter(hxs, hys, s=ns + 120, facecolors="none",
                   edgecolors=highlight_color, linewidths=2.2, zorder=4)

    _clean_ax(ax)
    if title:
        ax.set_title(title, fontsize=10)


def draw_graph_duals(
    graph: nx.Graph,
    duals: np.ndarray,
    layout: Dict[int, Tuple[float, float]],
    *,
    ax=None,
    title: Optional[str] = None,
    show_values: bool = True,
):
    """Draw a graph where node size and color encode dual weights."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(5.8, 4.0), dpi=110, constrained_layout=True)
    else:
        fig = ax.figure

    duals_arr = np.asarray(duals, dtype=np.float64)
    ymax = float(duals_arr.max()) if duals_arr.size else 1.0
    ymax = ymax or 1.0

    node_sizes: List[float] = []
    node_colors: List[Any] = []
    for v in graph.nodes():
        y = float(duals_arr[v]) if v < len(duals_arr) else 0.0
        if y <= 1e-9:
            node_sizes.append(120.0)
            node_colors.append(GREY)
        else:
            t = y / ymax
            node_sizes.append(200.0 + 600.0 * t)
            node_colors.append(_mix(QCI_TEAL, QCI_ORANGE, t))

    draw_graph(ax, graph, layout, node_color=node_colors, node_size=node_sizes,
               title=title)

    if show_values:
        for v, p in layout.items():
            y = float(duals_arr[v]) if v < len(duals_arr) else 0.0
            if y > 0.01:
                ax.text(p[0], p[1] - 0.12, f"{y:.2f}", ha="center", va="top",
                        fontsize=6, color=DARK_GREY)
    return fig, ax


def draw_columns_grid(
    graph: nx.Graph,
    columns: List[Iterable[int]],
    layout: Dict[int, Tuple[float, float]],
    *,
    ncols: int = 4,
    suptitle: Optional[str] = None,
    palette: Optional[List[str]] = None,
    figsize_per_col: Tuple[float, float] = (2.0, 1.9),
):
    """Small-multiples view: one subplot per column, IS highlighted."""
    import matplotlib.pyplot as plt

    palette = palette or IS_PALETTE
    cols = [list(c) for c in columns]
    count = len(cols)

    if count == 0:
        fig, ax = plt.subplots(figsize=(5.5, 3.0), constrained_layout=True)
        ax.text(0.5, 0.5, "(no profitable columns)", ha="center", va="center",
                fontsize=12, color=DARK_GREY)
        _clean_ax(ax)
        if suptitle:
            ax.set_title(suptitle, fontsize=11)
        return fig, [ax]

    ncols_eff = min(ncols, count)
    nrows = int(np.ceil(count / ncols_eff))
    fig, axes = plt.subplots(
        nrows, ncols_eff,
        figsize=(figsize_per_col[0] * ncols_eff, figsize_per_col[1] * nrows),
        dpi=110, constrained_layout=True,
    )
    axes = np.atleast_2d(axes).reshape(nrows, ncols_eff)

    for k in range(nrows * ncols_eff):
        r, c = divmod(k, ncols_eff)
        ax = axes[r, c]
        if k < count:
            col = cols[k]
            color = palette[k % len(palette)]
            node_colors = [color if v in col else "white" for v in graph.nodes()]
            draw_graph(ax, graph, layout, node_color=node_colors, node_size=180,
                       edge_alpha=0.25)
            ax.set_title(f"|IS|={len(col)}", fontsize=8, color=DARK_GREY)
        else:
            ax.set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    return fig, axes


def draw_coloring(
    graph: nx.Graph,
    coloring: Iterable[Iterable[int]],
    layout: Dict[int, Tuple[float, float]],
    *,
    ax=None,
    title: Optional[str] = None,
    palette: Optional[List[str]] = None,
):
    """Draw a vertex coloring, one color per IS."""
    import matplotlib.pyplot as plt

    palette = palette or IS_PALETTE
    color_classes = [list(cc) for cc in coloring]
    color_of = {}
    for k, cc in enumerate(color_classes):
        for v in cc:
            color_of[int(v)] = palette[k % len(palette)]

    if ax is None:
        fig, ax = plt.subplots(figsize=(5.0, 4.0), dpi=110, constrained_layout=True)
    else:
        fig = ax.figure

    node_colors = [color_of.get(v, GREY) for v in graph.nodes()]
    draw_graph(ax, graph, layout, node_color=node_colors, node_size=260,
               title=title or f"Coloring  (k={len(color_classes)})")
    return fig, ax


def kamada_kawai_layout(graph: nx.Graph, seed: int = 0) -> Dict[int, Tuple[float, float]]:
    """Stable Kamada-Kawai layout (used when a graph has no bundled layout)."""
    rng_state = np.random.get_state()
    try:
        np.random.seed(seed)
        pos = nx.kamada_kawai_layout(graph)
    finally:
        np.random.set_state(rng_state)
    return {int(v): (float(p[0]), float(p[1])) for v, p in pos.items()}


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def summarize_columns(
    columns: Iterable[Iterable[int]],
    dual_vars: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    cols = [list(c) for c in columns]
    sizes = [len(c) for c in cols]
    out: Dict[str, Any] = {
        "count": len(cols),
        "avg_size": float(np.mean(sizes)) if sizes else 0.0,
        "max_size": int(max(sizes)) if sizes else 0,
        "sizes": sizes,
    }
    if dual_vars is not None:
        d = np.asarray(dual_vars, dtype=np.float64)
        out["dual_sums"] = [float(sum(d[v] for v in c)) for c in cols]
    return out


def compare_oracle_calls_table(results: Dict[str, Dict[str, Any]]):
    """Produce a comparison DataFrame of solver runs.

    Each value in ``results`` is a dict with keys like:
      chi, runtime_s, iterations, columns_per_call, api_calls, notes.
    Missing keys appear as NaN. Falls back to a list-of-dicts if pandas is
    unavailable.
    """
    cols = ["chi", "runtime_s", "iterations", "columns_per_call",
            "api_calls", "notes"]
    rows = []
    for solver, stats in results.items():
        row = {"solver": solver}
        for c in cols:
            row[c] = stats.get(c)
        rows.append(row)
    try:
        import pandas as pd
        return pd.DataFrame(rows, columns=["solver", *cols])
    except ImportError:
        return rows


# ---------------------------------------------------------------------------
# Environment / credentials
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Direct-backend workaround: the main package's `_direct_solve_qp` calls
# `client.wait_for_lock()` but discards the returned lock_id, then submits the
# job with `lock_id=''` which the device rejects with err_code=4. This patch
# captures the lock_id, threads it through `solve_sum_constrained`, and
# releases it correctly. Applied lazily when first direct oracle is built.
# ---------------------------------------------------------------------------

_DIRECT_PATCH_INSTALLED = False


def _install_direct_workarounds() -> None:
    global _DIRECT_PATCH_INSTALLED
    if _DIRECT_PATCH_INSTALLED:
        return
    try:
        from quantum_colgen.pricing import dirac_oracle as _do
    except ImportError:
        return  # eqc-models stack not installed; nothing to patch

    def _patched_direct_solve_qp(
        linear, quadratic, num_samples, relax_schedule, sum_constraint,
        solution_precision, direct_config,
    ):
        from eqc_direct.client import EqcClient
        client = EqcClient(**direct_config)
        # Touch system_status (best-effort) like the original code did.
        ss = getattr(client, "system_status", None)
        if callable(ss):
            try:
                ss()
            except Exception:
                pass

        lock_id = ""
        try:
            wait = getattr(client, "wait_for_lock", None)
            if callable(wait):
                result = wait()
                if isinstance(result, tuple) and result \
                        and isinstance(result[0], str):
                    lock_id = result[0]

            poly_indices, poly_coefficients = _do._build_polynomial_terms(
                linear, quadratic
            )
            payload = client.solve_sum_constrained(
                poly_indices=poly_indices,
                poly_coefficients=poly_coefficients,
                num_variables=int(len(linear)),
                num_samples=int(num_samples),
                relaxation_schedule=int(relax_schedule),
                sum_constraint=int(sum_constraint),
                solution_precision=solution_precision,
                lock_id=lock_id,
            )
            # Bug #3: `_direct_extract_solutions` only checks 'solutions'
            # (plural) but eqc-direct returns 'solution' (singular).
            # Try our key first, then fall back to the package's normaliser.
            if isinstance(payload, dict) and payload.get("solution"):
                vectors = payload["solution"]
                return [np.asarray(v, dtype=np.float64) for v in vectors]
            return _do._direct_extract_solutions(payload)
        except _do.DirectDiracBackendError:
            raise
        except Exception as exc:
            raise _do.DirectDiracBackendError(
                f"Direct Dirac solve failed: {exc}"
            ) from exc
        finally:
            if lock_id:
                try:
                    client.release_lock(lock_id=lock_id)
                except Exception:
                    pass

    _do._direct_solve_qp = _patched_direct_solve_qp
    _DIRECT_PATCH_INSTALLED = True


# Default endpoints documented in ~/Code/qci/skills/max-clique-skills/.env.example.
# Note: the skill .env.example lists 172.18.41.79, but live network probing
# (May 2026) shows that endpoint is offline; 172.18.41.228 is the current
# working Dirac-3 hardware address.
DEFAULT_QCI_API_URL = "https://api.qci-prod.com"
DEFAULT_DIRECT_IP_ADDRESS = "172.18.41.228"
DEFAULT_DIRECT_PORT = "50051"


def _candidate_dotenv_paths() -> List[Path]:
    """Order: project-local first, then shared QCi skills .env files."""
    home = Path.home()
    return [
        REPO_ROOT / ".env",
        REPO_ROOT / "quantum-branch-price" / ".env",
        home / "Code" / "qci" / "skills" / ".env",
        home / "Code" / "qci" / "skills" / "qci-eqc-models" / ".env",
        home / "Code" / "qci" / "skills" / "max-clique-skills" / ".env",
    ]


def load_dotenv_if_present(*candidates: Path) -> bool:
    """Best-effort .env loader. Returns True if any file was loaded.

    Search order: project-local ``.env`` files, then the shared QCi skills
    ``.env`` files under ``~/Code/qci/skills/`` (which carry the cloud
    `QCI_API_URL` + `QCI_TOKEN` and the direct hardware endpoint). Existing
    env vars are never overwritten. Lines look like ``KEY=value``.
    """
    paths = [*_candidate_dotenv_paths(), *candidates]
    loaded = False
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
                loaded = True
    return loaded


def prompt_cloud_credentials(force: bool = False) -> None:
    """Prompt for QCI cloud credentials interactively if not already set.

    Sets both ``QCI_TOKEN`` and ``QCI_API_URL`` (the cloud solver in
    ``eqc_models`` requires both). Falls back to ``https://api.qci-prod.com``
    if the user provides no URL.
    """
    if force or not os.environ.get("QCI_TOKEN"):
        token = getpass.getpass("QCI_TOKEN (input hidden): ").strip()
        if token:
            os.environ["QCI_TOKEN"] = token
    if force or not os.environ.get("QCI_API_URL"):
        url = input(f"QCI_API_URL [default {DEFAULT_QCI_API_URL}]: ").strip()
        os.environ["QCI_API_URL"] = url or DEFAULT_QCI_API_URL


def prompt_direct_credentials(force: bool = False) -> Dict[str, Optional[str]]:
    """Prompt for eqc-direct connection settings; returns a dict of the values.

    Sets the ``EQC_DIRECT_IP_ADDRESS`` / ``EQC_DIRECT_PORT`` /
    ``EQC_DIRECT_CERT_FILE`` env vars (which the main package's
    ``DiracPricingOracle`` reads via ``_resolve_direct_config``). The
    skill's documented defaults are IP ``172.18.41.79`` and port ``50051``.
    """
    out: Dict[str, Optional[str]] = {}
    if force or not os.environ.get("EQC_DIRECT_IP_ADDRESS"):
        ip = input(
            f"EQC_DIRECT_IP_ADDRESS [default {DEFAULT_DIRECT_IP_ADDRESS}]: "
        ).strip()
        os.environ["EQC_DIRECT_IP_ADDRESS"] = ip or DEFAULT_DIRECT_IP_ADDRESS
        out["ip_address"] = os.environ["EQC_DIRECT_IP_ADDRESS"]
    else:
        out["ip_address"] = os.environ["EQC_DIRECT_IP_ADDRESS"]

    if force or not os.environ.get("EQC_DIRECT_PORT"):
        port = input(
            f"EQC_DIRECT_PORT [default {DEFAULT_DIRECT_PORT}]: "
        ).strip()
        os.environ["EQC_DIRECT_PORT"] = port or DEFAULT_DIRECT_PORT
        out["port"] = os.environ["EQC_DIRECT_PORT"]
    else:
        out["port"] = os.environ["EQC_DIRECT_PORT"]

    cert = input("EQC_DIRECT_CERT_FILE (blank if none): ").strip()
    if cert:
        os.environ["EQC_DIRECT_CERT_FILE"] = cert
        out["cert_file"] = cert
    return out


# ---------------------------------------------------------------------------
# Sample I/O — local copies of the rf_branching sample_io schema, kept in
# this module so notebooks don't depend on the qbp/rf_branching packages.
# Schema is binary-compatible with RF-branching/src/rf_branching/sample_io.py.
# ---------------------------------------------------------------------------

import hashlib
import tempfile

SCHEMA_VERSION = 1


def _canonical_edges(edges: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    return sorted((min(u, v), max(u, v)) for u, v in edges)


def graph_signature(node_list: List[int], edges: Iterable[Tuple[int, int]]) -> str:
    h = hashlib.sha256()
    h.update(b"N:")
    h.update(",".join(str(x) for x in sorted(node_list)).encode())
    h.update(b"\nE:")
    for u, v in _canonical_edges(edges):
        h.update(f"{u} {v}\n".encode())
    return h.hexdigest()


def dual_signature(node_list: List[int], dual_vars: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(dual_vars, dtype=np.float64))
    h = hashlib.sha256()
    h.update(b"N:")
    h.update(",".join(str(x) for x in node_list).encode())
    h.update(b"\nD:")
    h.update(arr.tobytes())
    return h.hexdigest()


@dataclass
class CallRecord:
    """Subset of rf_branching.sample_io.CallRecord we need locally."""

    schema_version: int
    call_idx: int
    timestamp_utc: str
    node_list: List[int]
    graph_edges: List[Tuple[int, int]]
    dual_vars: np.ndarray
    dual_by_label: Dict[int, float]
    solver_C: np.ndarray
    solver_J: np.ndarray
    solver_params: Dict[str, Any]
    internal_node_list: List[int]
    raw_solutions: np.ndarray
    device_seconds: float
    api_seconds: float
    extract_seconds: float
    columns_found: int
    graph_sig: str
    dual_sig: str
    method: str
    oracle_config: Dict[str, Any]
    latest_sample_telemetry: Optional[Dict[str, Any]] = None
    node_id: Optional[int] = None
    cg_iter: Optional[int] = None


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=path.suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _index_entry(rec: CallRecord, pkl_relpath: str) -> Dict[str, Any]:
    return {
        "call_idx": rec.call_idx,
        "pkl": pkl_relpath,
        "node_id": rec.node_id,
        "cg_iter": rec.cg_iter,
        "graph_sig": rec.graph_sig,
        "dual_sig": rec.dual_sig,
        "subgraph_n": len(rec.node_list),
        "subgraph_m": len(rec.graph_edges),
        "internal_n": len(rec.internal_node_list),
        "num_raw_solutions": int(rec.raw_solutions.shape[0]) if rec.raw_solutions.size else 0,
        "api_s": rec.api_seconds,
        "device_s": rec.device_seconds,
        "extract_s": rec.extract_seconds,
        "columns_found": rec.columns_found,
        "method": rec.method,
        "timestamp_utc": rec.timestamp_utc,
    }


def save_call_record(raw_dir: Path, rec: CallRecord) -> Path:
    raw_dir = Path(raw_dir)
    pkl_path = raw_dir / f"call_{rec.call_idx:05d}.pkl"
    _atomic_write_bytes(pkl_path, pickle.dumps(rec, protocol=pickle.HIGHEST_PROTOCOL))
    idx_path = raw_dir / "index.jsonl"
    with open(idx_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(_index_entry(rec, pkl_path.name)) + "\n")
    return pkl_path


def load_call_index(raw_dir: Path) -> List[Dict[str, Any]]:
    raw_dir = Path(raw_dir)
    idx_path = raw_dir / "index.jsonl"
    if not idx_path.exists():
        return []
    out = []
    for line in idx_path.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    out.sort(key=lambda r: r["call_idx"])
    return out


def load_call_record(raw_dir: Path, call_idx: int):
    """Load a CallRecord pickle, tolerating either local or rf_branching schema."""
    pkl_path = Path(raw_dir) / f"call_{call_idx:05d}.pkl"
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Replay shim — works against the main quantum_colgen DiracPricingOracle by
# monkey-patching the module-level _dirac_solve_qp() to return saved
# raw_solutions instead of contacting Dirac. Same idea as
# rf_branching.replay_oracle.ReplayDiracOracle, but self-contained.
# ---------------------------------------------------------------------------

def _import_main_dirac():
    from quantum_colgen.pricing import dirac_oracle as _do
    from quantum_colgen.pricing.dirac_oracle import DiracPricingOracle
    return _do, DiracPricingOracle


def _make_fake_cloud_solver(get_next_record):
    """Build a stand-in for Dirac3ContinuousCloudSolver.

    ``get_next_record`` is a 0-arg callable that pops the next saved
    CallRecord-equivalent (must expose ``raw_solutions`` ndarray).
    """

    class FakeSolver:
        def __init__(self, *a: Any, **k: Any) -> None:
            self.client = None

        def solve(self, model, **params):  # noqa: D401
            rec = get_next_record()
            solutions_as_lists = [list(map(float, row)) for row in rec.raw_solutions]
            return {
                "results": {"solutions": solutions_as_lists},
                "job_info": {"job_id": "replay"},
            }

    return FakeSolver


class ReplayDiracOracle:  # noqa: D401 — replays cached calls
    """Replays recorded Dirac responses against the main package.

    Constructed lazily so that simply importing _demo_utils does not require
    eqc-models / qci-client (the heavy cloud deps). Constructing this class
    *does* require those deps because it subclasses DiracPricingOracle.
    Patches both ``_dirac_solve_qp`` (used by ``method='filter'``) and
    ``Dirac3ContinuousCloudSolver`` (used directly by ``method='gibbons'``).
    """

    def __new__(cls, raw_dir: Path, *, strict: bool = True, **kwargs: Any):
        _do, _Base = _import_main_dirac()

        class _Impl(_Base):
            def __init__(inner, raw_dir: Path, strict: bool, **kw: Any) -> None:
                super().__init__(**kw)
                inner._raw_dir = Path(raw_dir)
                inner._index = load_call_index(inner._raw_dir)
                inner._cursor = 0
                inner._strict = strict

            def remaining(inner) -> int:
                return max(0, len(inner._index) - inner._cursor)

            def reset_cursor(inner) -> None:
                inner._cursor = 0

            def _peek_next(inner, expected_gs: str, expected_ds: str):
                if inner._cursor >= len(inner._index):
                    raise RuntimeError(
                        f"ReplayDiracOracle: cursor={inner._cursor} exceeds "
                        f"recorded calls={len(inner._index)} (no more samples)."
                    )
                entry = inner._index[inner._cursor]
                if inner._strict and (
                    entry["graph_sig"] != expected_gs
                    or entry["dual_sig"] != expected_ds
                ):
                    raise RuntimeError(
                        f"Replay signature mismatch at call {entry['call_idx']}:\n"
                        f"  recorded graph={entry['graph_sig'][:8]} "
                        f"vs current={expected_gs[:8]}\n"
                        f"  recorded dual={entry['dual_sig'][:8]} "
                        f"vs current={expected_ds[:8]}"
                    )
                rec = load_call_record(inner._raw_dir, entry["call_idx"])
                inner._cursor += 1
                return rec

            def solve(inner, graph: nx.Graph, dual_vars: np.ndarray) -> List[Set[int]]:
                outer_nodes = sorted(graph.nodes())
                outer_edges = list(graph.edges())
                expected_gs = graph_signature(outer_nodes, outer_edges)
                expected_ds = dual_signature(
                    outer_nodes, np.asarray(dual_vars, dtype=np.float64)
                )

                def get_next():
                    return inner._peek_next(expected_gs, expected_ds)

                def fake_qp(adjacency_matrix, **_kw):
                    rec = get_next()
                    return [np.array(s, dtype=np.float64) for s in rec.raw_solutions]

                FakeSolver = _make_fake_cloud_solver(get_next)

                with mock.patch.object(_do, "_dirac_solve_qp", fake_qp), \
                     mock.patch.object(_do, "Dirac3ContinuousCloudSolver", FakeSolver):
                    return _Base.solve(inner, graph, dual_vars)

        return _Impl(raw_dir=raw_dir, strict=strict, **kwargs)


# ---------------------------------------------------------------------------
# Capturing oracle — saves every call to a run directory so that the same
# pipeline can be replayed offline later. Uses the local CallRecord schema.
# ---------------------------------------------------------------------------

class CapturingPricingOracle:
    """Wrap any PricingOracle and persist each call to ``run_dir/raw_samples/``.

    Patches the module-level ``_dirac_solve_qp`` so the raw solution vectors
    can be intercepted (the inner oracle still does extraction normally —
    the wrapper just grabs the vectors flowing through). Falls back to an
    empty raw_solutions array if the inner oracle is not Dirac-based.
    """

    def __init__(self, inner, run_dir: Path) -> None:
        self.inner = inner
        self.run_dir = Path(run_dir)
        self.raw_dir = self.run_dir / "raw_samples"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self._call_idx = len(load_call_index(self.raw_dir))
        # mirror the inner oracle's timer for orchestrators that look it up
        self.timer = getattr(inner, "timer", None)
        self._is_dirac = self._inner_is_dirac()

    def _inner_is_dirac(self) -> bool:
        try:
            from quantum_colgen.pricing.dirac_oracle import DiracPricingOracle
            return isinstance(self.inner, DiracPricingOracle)
        except Exception:
            return False

    def solve(self, graph: nx.Graph, dual_vars: np.ndarray) -> List[Set[int]]:
        outer_nodes = sorted(graph.nodes())
        outer_edges = [(int(u), int(v)) for u, v in graph.edges()]
        dv = np.asarray(dual_vars, dtype=np.float64)

        captured: Dict[str, Any] = {
            "raw": None,
            "internal_n": 0,
            "device_s": 0.0,
        }

        if self._is_dirac:
            from quantum_colgen.pricing import dirac_oracle as _do
            real_qp = _do._dirac_solve_qp
            real_cloud_cls = getattr(_do, "Dirac3ContinuousCloudSolver", None)

            def capturing_qp(adjacency_matrix, **kwargs):
                solutions = real_qp(adjacency_matrix, **kwargs)
                if solutions is not None and len(solutions) > 0:
                    captured["raw"] = np.asarray(solutions, dtype=np.float64)
                    captured["internal_n"] = int(adjacency_matrix.shape[0])
                return solutions

            class CapturingCloudSolver:
                """Thin proxy around the real cloud solver that snapshots the response."""

                def __init__(self, *a, **k):
                    self._real = real_cloud_cls(*a, **k) if real_cloud_cls else None

                def solve(self, model, **params):
                    response = self._real.solve(model, **params)
                    sols = response.get("results", {}).get("solutions", []) if response else []
                    if sols:
                        captured["raw"] = np.array(sols, dtype=np.float64)
                        captured["internal_n"] = int(captured["raw"].shape[1])
                    return response

            t0 = time.monotonic()
            patches = [mock.patch.object(_do, "_dirac_solve_qp", capturing_qp)]
            if real_cloud_cls is not None:
                patches.append(
                    mock.patch.object(_do, "Dirac3ContinuousCloudSolver", CapturingCloudSolver)
                )
            try:
                for p in patches:
                    p.start()
                result = self.inner.solve(graph, dv)
            finally:
                for p in reversed(patches):
                    p.stop()
            wall = time.monotonic() - t0
        else:
            t0 = time.monotonic()
            result = self.inner.solve(graph, dv)
            wall = time.monotonic() - t0

        if captured["raw"] is None:
            # Either non-Dirac inner, or the oracle short-circuited before calling Dirac.
            return result

        # api_seconds estimation: if the inner oracle has a timer, use the latest call;
        # else fall back to wall.
        api_s = wall
        ext_s = 0.0
        if self.timer is not None and getattr(self.timer, "calls", None):
            last = self.timer.calls[-1]
            api_s = float(getattr(last, "api_seconds", api_s))
            ext_s = float(getattr(last, "extract_seconds", 0.0))

        rec = CallRecord(
            schema_version=SCHEMA_VERSION,
            call_idx=self._call_idx,
            timestamp_utc=datetime.now(timezone.utc)
                .strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            node_list=outer_nodes,
            graph_edges=outer_edges,
            dual_vars=np.ascontiguousarray(dv),
            dual_by_label={int(n): float(dv[i]) for i, n in enumerate(outer_nodes)},
            solver_C=np.zeros(captured["internal_n"], dtype=np.float64),
            solver_J=np.zeros(
                (captured["internal_n"], captured["internal_n"]), dtype=np.float64
            ),
            solver_params={
                "num_samples": getattr(self.inner, "num_samples", None),
                "method": getattr(self.inner, "method", None),
                "backend": getattr(self.inner, "backend", None),
            },
            internal_node_list=list(range(captured["internal_n"])),
            raw_solutions=captured["raw"],
            device_seconds=0.0,
            api_seconds=api_s,
            extract_seconds=ext_s,
            columns_found=len(result),
            graph_sig=graph_signature(outer_nodes, outer_edges),
            dual_sig=dual_signature(outer_nodes, dv),
            method=str(getattr(self.inner, "method", "gibbons")),
            oracle_config={
                "support_thresholds": list(getattr(self.inner, "support_thresholds", [])),
                "multi_prune": bool(getattr(self.inner, "multi_prune", False)),
                "num_random_prune_trials": int(
                    getattr(self.inner, "num_random_prune_trials", 3)
                ),
                "randomized_rounding": bool(getattr(self.inner, "randomized_rounding", False)),
                "num_random_rounds": int(getattr(self.inner, "num_random_rounds", 10)),
                "local_search_passes": int(getattr(self.inner, "local_search_passes", 0)),
            },
        )
        save_call_record(self.raw_dir, rec)
        self._call_idx += 1
        return result


def new_run_dir(label: str) -> Path:
    """Create ``notebooks/runs/<UTC-timestamp>_<label>/`` and return the path."""
    NOTEBOOK_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = NOTEBOOK_RUNS_DIR / f"{ts}_{label}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_run_metadata(run_dir: Path, graph: nx.Graph, config: Dict[str, Any]) -> None:
    run_dir = Path(run_dir)
    meta = {
        "n": graph.number_of_nodes(),
        "m": graph.number_of_edges(),
        "nodes": sorted(graph.nodes()),
        "edges": [[int(u), int(v)] for u, v in graph.edges()],
        "config": config,
        "timestamp_utc": datetime.now(timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
    }
    (run_dir / "graph.json").write_text(json.dumps(meta, indent=2))


def replay_oracle_from_run(run_dir: Path, **overrides: Any) -> "ReplayDiracOracle":
    """Reconstruct a replay oracle from a previously-saved run directory.

    Auto-loads ``method`` and the extraction settings (``multi_prune``,
    ``randomized_rounding``, ``num_random_rounds``, ``local_search_passes``,
    ``support_thresholds``) from the first saved call record so the replay
    reproduces the same independent sets as the live run. Caller can
    override any of these by passing them explicitly.
    """
    raw_dir = Path(run_dir) / "raw_samples"
    if not raw_dir.exists():
        raise FileNotFoundError(f"No raw_samples dir under {run_dir}")
    index = load_call_index(raw_dir)
    if not index:
        raise RuntimeError(f"No call records found under {raw_dir}")
    rec = load_call_record(raw_dir, index[0]["call_idx"])

    # Pull live extraction config out of the first call's record.
    cfg = getattr(rec, "oracle_config", {}) or {}
    params = getattr(rec, "solver_params", {}) or {}

    auto = {
        "method": params.get("method") or "gibbons",
        "num_samples": params.get("num_samples") or 100,
        "multi_prune": cfg.get("multi_prune", False),
        "randomized_rounding": cfg.get("randomized_rounding", False),
        "num_random_rounds": cfg.get("num_random_rounds", 10),
        "num_random_prune_trials": cfg.get("num_random_prune_trials", 3),
        "local_search_passes": cfg.get("local_search_passes", 5),
    }
    if cfg.get("support_thresholds"):
        auto["support_thresholds"] = list(cfg["support_thresholds"])
    auto.update(overrides)
    return ReplayDiracOracle(raw_dir=raw_dir, backend="cloud", **auto)


# ---------------------------------------------------------------------------
# Backend factory — single-toggle entry point for notebooks
# ---------------------------------------------------------------------------

def make_dirac_oracle(
    mode: str = "replay",
    *,
    raw_dir: Optional[Path] = None,
    method: str = "gibbons",
    num_samples: int = 100,
    relax_schedule: int = 2,
    sum_constraint: int = 1,
    multi_prune: bool = True,
    randomized_rounding: bool = True,
    num_random_rounds: int = 10,
    random_seed: Optional[int] = 42,
    local_search_passes: int = 5,
    direct_ip_address: Optional[str] = None,
    direct_port: Optional[int] = None,
    direct_cert_file: Optional[str] = None,
    interactive: bool = True,
    replay_strict: bool = False,
    **extra: Any,
):
    """Return a Dirac pricing oracle for the chosen backend mode.

    mode:
      - "replay": uses ``raw_dir`` (defaults to the bundled
        RF-branching/instances/<DEFAULT_BUNDLED_INSTANCE>/<DEFAULT_BUNDLED_METHOD>/raw_samples).
      - "cloud":  queries the QCi cloud API. Prompts for QCI_TOKEN if missing
        when ``interactive=True``.
      - "direct": uses eqc-direct on-prem. Resolves EQC_DIRECT_* env vars or
        prompts when ``interactive=True``.
    """
    common = dict(
        method=method,
        num_samples=num_samples,
        relax_schedule=relax_schedule,
        sum_constraint=sum_constraint,
        multi_prune=multi_prune,
        randomized_rounding=randomized_rounding,
        num_random_rounds=num_random_rounds,
        random_seed=random_seed,
        local_search_passes=local_search_passes,
    )
    common.update(extra)

    if mode == "replay":
        rd = Path(raw_dir) if raw_dir else (
            RAW_SAMPLES_ROOT / DEFAULT_BUNDLED_INSTANCE / DEFAULT_BUNDLED_METHOD / "raw_samples"
        )
        if not rd.exists():
            raise FileNotFoundError(
                f"Bundled replay data not found at {rd}. Either pass raw_dir=... or check "
                "that the RF-branching submodule is initialised."
            )
        # Default replay_strict=False because the bundled samples were recorded by
        # the qbp package's CG; rerunning CG with the main package will pick
        # numerically-equivalent-but-different duals after the first few iters.
        # We still consume saved samples in order, so the Dirac responses are
        # genuine — only the (graph, duals) signature check is relaxed.
        return ReplayDiracOracle(
            raw_dir=rd, backend="cloud", strict=replay_strict, **common
        )

    if mode == "cloud":
        load_dotenv_if_present()
        # Cloud solver in eqc_models requires BOTH QCI_TOKEN and QCI_API_URL.
        if interactive and (
            not os.environ.get("QCI_TOKEN") or not os.environ.get("QCI_API_URL")
        ):
            prompt_cloud_credentials()
        elif not os.environ.get("QCI_API_URL"):
            os.environ["QCI_API_URL"] = DEFAULT_QCI_API_URL
        from quantum_colgen.pricing.dirac_oracle import DiracPricingOracle
        return DiracPricingOracle(backend="cloud", **common)

    if mode == "direct":
        load_dotenv_if_present()
        if interactive and not os.environ.get("EQC_DIRECT_IP_ADDRESS") \
                and direct_ip_address is None:
            prompt_direct_credentials()
        # Apply documented defaults so users on the QCi VPN don't need to
        # supply values explicitly when the skill defaults already match.
        if direct_ip_address is None and not os.environ.get("EQC_DIRECT_IP_ADDRESS"):
            direct_ip_address = DEFAULT_DIRECT_IP_ADDRESS
        if direct_port is None and not os.environ.get("EQC_DIRECT_PORT"):
            direct_port = int(DEFAULT_DIRECT_PORT)
        _install_direct_workarounds()
        from quantum_colgen.pricing.dirac_oracle import DiracPricingOracle
        oracle = DiracPricingOracle(
            backend="direct",
            direct_ip_address=direct_ip_address,
            direct_port=direct_port,
            direct_cert_file=direct_cert_file,
            **common,
        )
        # Workaround for bug #1: `_resolve_direct_config` coerces port to int,
        # but `eqc_direct.client.EqcClient` concatenates port as a string.
        if oracle._direct_config and "port" in oracle._direct_config:
            oracle._direct_config["port"] = str(oracle._direct_config["port"])
        return oracle

    raise ValueError(f"Unknown mode {mode!r} (expected replay|cloud|direct)")


# ---------------------------------------------------------------------------
# Recording-during-CG hook (for the iteration browser in notebook 02b)
# ---------------------------------------------------------------------------

class HistoryPricingOracle:
    """Wrap a pricing oracle and stash (graph, duals, columns) per call.

    Useful for the iteration browser in 02b. Composable with CapturingPricingOracle:
    wrap with HistoryPricingOracle on the outside if you want both in-memory
    history and on-disk capture.
    """

    def __init__(self, inner) -> None:
        self.inner = inner
        self.history: List[Dict[str, Any]] = []
        self.timer = getattr(inner, "timer", None)

    def solve(self, graph: nx.Graph, dual_vars: np.ndarray) -> List[Set[int]]:
        cols = self.inner.solve(graph, dual_vars)
        self.history.append({
            "node_list": sorted(graph.nodes()),
            "graph_edges": [(int(u), int(v)) for u, v in graph.edges()],
            "dual_vars": np.asarray(dual_vars, dtype=np.float64).copy(),
            "columns": [set(c) for c in cols],
        })
        return cols


__all__ = [
    # paths
    "REPO_ROOT", "NOTEBOOKS_DIR", "NOTEBOOK_DATA_DIR", "NOTEBOOK_RUNS_DIR",
    "SLIDES_DEMO_DIR", "SLIDES_DATA_DIR", "RAW_SAMPLES_ROOT",
    "DEFAULT_BUNDLED_INSTANCE", "DEFAULT_BUNDLED_METHOD",
    # palette
    "QCI_BLUE", "QCI_TEAL", "QCI_ORANGE", "QCI_GREEN", "GREY", "DARK_GREY",
    "IS_PALETTE",
    # data
    "load_psp", "psp_to_graph", "psp_to_layout", "psp_dual_array",
    # viz
    "draw_graph", "draw_graph_duals", "draw_columns_grid", "draw_coloring",
    "kamada_kawai_layout",
    # stats
    "summarize_columns", "compare_oracle_calls_table",
    # env
    "load_dotenv_if_present", "prompt_cloud_credentials", "prompt_direct_credentials",
    # capture / replay
    "graph_signature", "dual_signature", "save_call_record", "load_call_index",
    "load_call_record", "CallRecord", "ReplayDiracOracle", "CapturingPricingOracle",
    "HistoryPricingOracle", "new_run_dir", "save_run_metadata",
    "replay_oracle_from_run",
    # backend
    "make_dirac_oracle",
]
