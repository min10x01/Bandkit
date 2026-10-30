"""bandkit: VASP band-structure parsing, analysis and plotting."""

from .analysis import *  # noqa: F401,F403
from .analysis import __all__ as _analysis_all
from .core import BandStructure
from .io import load_band_structure

__version__ = "0.1.0"
__all__ = ["BandStructure", "load_band_structure", *_analysis_all]
