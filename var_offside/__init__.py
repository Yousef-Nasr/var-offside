"""
VAR-Offside: offside detection for football broadcast frames.

    from var_offside import OffsidePipeline

See https://yousef-nasr.github.io/var-offside/ for the documentation.
"""

__version__ = "0.1.0"

from var_offside.pipeline import OffsidePipeline

__all__ = ["OffsidePipeline", "__version__"]
