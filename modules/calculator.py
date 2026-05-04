"""
MACE calculator setup and device detection.
"""

from __future__ import annotations

from typing import Literal

import streamlit as st

ModelSize = Literal["small", "medium", "large"]
DType = Literal["float32", "float64"]

# --- optional torch import ---------------------------------------------------
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# --- optional mace import (needs matscipy which requires C compiler) ---------
try:
    from mace.calculators import mace_mp as _mace_mp  # noqa: F401
    _MACE_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    _MACE_AVAILABLE = False


def detect_device() -> str:
    """Return ``'cuda'`` if a GPU is available, otherwise ``'cpu'``."""
    if _TORCH_AVAILABLE:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


def is_mace_available() -> bool:
    """Return True if the MACE package was imported successfully."""
    return _MACE_AVAILABLE


@st.cache_resource(show_spinner="Loading MACE model…")
def get_calculator(
    model: str,
    device: str,
    default_dtype: DType,
):
    """Load and cache a MACE-MP calculator.

    Uses :func:`streamlit.cache_resource` so the model weights are only
    loaded once per session, regardless of how many times the function is
    called with the same arguments.

    Parameters
    ----------
    model:
        One of ``'small'``, ``'medium'``, ``'large'``, or an absolute
        path to a custom ``.model`` file.
    device:
        PyTorch device string: ``'cpu'`` or ``'cuda'``.
    default_dtype:
        Floating-point precision: ``'float32'`` or ``'float64'``.

    Returns
    -------
    MACECalculator
        Configured MACE calculator instance ready for use with ASE.

    Raises
    ------
    RuntimeError
        If MACE or its dependencies are not installed.
    """
    if not _MACE_AVAILABLE:
        raise RuntimeError(
            "MACE is not available in this environment.\n\n"
            "On Windows, the easiest way to install it is via conda:\n"
            "  conda install -c conda-forge mace-torch\n\n"
            "Alternatively, install Microsoft C++ Build Tools and then:\n"
            "  pip install mace-torch"
        )

    from mace.calculators import mace_mp  # re-import for clarity

    return mace_mp(
        model=model,
        device=device,
        default_dtype=default_dtype,
    )
