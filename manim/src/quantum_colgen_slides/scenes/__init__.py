"""Slide scenes for the presentation."""

from .s01_title import TitleSlide
from .s02_graph_coloring import GraphColoringSlide
from .s03_independent_set import IndependentSetSlide
from .s04_cg_overview import CGOverviewSlide
from .s05_rmp import RMPSlide
from .s06_psp import PSPSlide
from .s07_milp_limitation import MILPLimitationSlide
from .s08_quantum_intro import QuantumIntroSlide
from .s09_dirac_extraction import DiracExtractionSlide
from .s10_benchmark import BenchmarkSlide
from .s10a_er40_graph import ER40GraphSlide
from .s10b_er40_colorings import ER40ColoringsSlide
from .s10c_er50_graph import ER50GraphSlide
from .s10d_er50_colorings import ER50ColoringsSlide
from .s11_summary import SummarySlide

__all__ = [
    "TitleSlide",
    "GraphColoringSlide",
    "IndependentSetSlide",
    "CGOverviewSlide",
    "RMPSlide",
    "PSPSlide",
    "MILPLimitationSlide",
    "QuantumIntroSlide",
    "DiracExtractionSlide",
    "BenchmarkSlide",
    "ER40GraphSlide",
    "ER40ColoringsSlide",
    "ER50GraphSlide",
    "ER50ColoringsSlide",
    "SummarySlide",
]
