"""
Structure loading, classification, and analysis utilities.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
import ase.io

import config


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_structure(file_bytes: bytes, filename: str) -> Atoms:
    """Load an atomic structure from raw file bytes.

    Supports POSCAR/CONTCAR (no extension or .vasp), CIF, XYZ,
    Extended XYZ, and ASE trajectory formats.

    Parameters
    ----------
    file_bytes:
        Raw bytes from the uploaded file.
    filename:
        Original filename used to infer the format.

    Returns
    -------
    Atoms
        ASE Atoms object representing the last (or only) structure frame.
    """
    suffix = Path(filename).suffix.lower()
    stem = Path(filename).stem.upper()

    # Map file extensions / stem names to ASE format strings
    _ext_map: dict[str, str] = {
        ".vasp": "vasp",
        ".poscar": "vasp",
        ".cif": "cif",
        ".xyz": "xyz",
        ".extxyz": "extxyz",
        ".traj": "traj",
    }
    _stem_map: dict[str, str] = {
        "POSCAR": "vasp",
        "CONTCAR": "vasp",
        "OUTCAR": "vasp",
    }

    fmt: str | None = _stem_map.get(stem) or _ext_map.get(suffix)

    with tempfile.NamedTemporaryFile(
        suffix=suffix or ".tmp", delete=False
    ) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        if fmt == "traj":
            # Read the last frame of the trajectory
            atoms = ase.io.read(tmp_path, index=-1, format="traj")
        elif fmt:
            atoms = ase.io.read(tmp_path, format=fmt)
        else:
            # Let ASE guess the format
            atoms = ase.io.read(tmp_path)
    finally:
        os.unlink(tmp_path)

    return atoms  # type: ignore[return-value]


def load_all_frames(file_bytes: bytes, filename: str) -> list[Atoms]:
    """Load all frames from a trajectory or multi-frame file.

    Parameters
    ----------
    file_bytes:
        Raw bytes from the uploaded file.
    filename:
        Original filename.

    Returns
    -------
    list[Atoms]
        All frames read from the file.
    """
    suffix = Path(filename).suffix.lower()
    fmt = "traj" if suffix == ".traj" else None

    with tempfile.NamedTemporaryFile(
        suffix=suffix or ".tmp", delete=False
    ) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        if fmt:
            frames = ase.io.read(tmp_path, index=":", format=fmt)
        else:
            frames = ase.io.read(tmp_path, index=":")
        # ase.io.read with index=":" always returns a list
        if isinstance(frames, Atoms):
            frames = [frames]
    finally:
        os.unlink(tmp_path)

    return list(frames)


def classify_structure(
    atoms: Atoms,
    vacuum_threshold: float = config.VACUUM_THRESHOLD,
) -> str:
    """Classify a structure as 'bulk' or 'slab'.

    A slab is identified when the c-axis of the cell is significantly
    larger than the atomic extent in z, indicating a vacuum region.

    Parameters
    ----------
    atoms:
        ASE Atoms object.
    vacuum_threshold:
        Minimum vacuum gap (Å) required to classify as a slab.

    Returns
    -------
    str
        ``'slab'`` or ``'bulk'``.
    """
    cell = atoms.get_cell()
    c_length = float(np.linalg.norm(cell[2]))
    positions = atoms.get_positions()
    z_extent = float(positions[:, 2].max() - positions[:, 2].min())
    vacuum = c_length - z_extent
    return "slab" if vacuum >= vacuum_threshold else "bulk"


def suggest_parameters(structure_type: str) -> dict[str, Any]:
    """Return smart default simulation parameters for a given structure type.

    Parameters
    ----------
    structure_type:
        ``'bulk'`` or ``'slab'``.

    Returns
    -------
    dict
        Suggested parameter values as a flat dictionary.
    """
    if structure_type == "slab":
        return {
            "pressure_gpa": 0.0,
            "relax_cell": False,
            "fmax_coarse": 0.1,
            "fmax_fine": 0.01,
        }
    # bulk defaults
    return {
        "pressure_gpa": 0.0,
        "relax_cell": True,
        "fmax_coarse": 0.1,
        "fmax_fine": 0.01,
    }


def apply_constraints(atoms: Atoms) -> Atoms:
    """Apply FixAtoms constraints based on Selective Dynamics flags.

    For POSCAR files with Selective Dynamics, ASE stores the per-atom
    flags in ``atoms.arrays['selective_dynamics']``.  Atoms where all
    three flags are ``False`` (i.e. fully frozen) are fixed.

    Parameters
    ----------
    atoms:
        ASE Atoms object (modified in-place).

    Returns
    -------
    Atoms
        The same Atoms object with constraints applied.
    """
    if "selective_dynamics" in atoms.arrays:
        sd = atoms.arrays["selective_dynamics"]
        fixed_indices = [i for i, flags in enumerate(sd) if not any(flags)]
        if fixed_indices:
            atoms.set_constraint(FixAtoms(indices=fixed_indices))
    return atoms


def get_structure_summary(atoms: Atoms) -> dict[str, Any]:
    """Extract key structural information for display.

    Parameters
    ----------
    atoms:
        ASE Atoms object.

    Returns
    -------
    dict
        Keys: ``formula``, ``n_atoms``, ``cell``, ``species``,
        ``has_constraints``, ``pbc``.
    """
    cell = atoms.get_cell()
    return {
        "formula": atoms.get_chemical_formula(),
        "n_atoms": len(atoms),
        "cell": cell.array.tolist(),
        "species": sorted(set(atoms.get_chemical_symbols())),
        "has_constraints": len(atoms.constraints) > 0,
        "pbc": atoms.get_pbc().tolist(),
    }
