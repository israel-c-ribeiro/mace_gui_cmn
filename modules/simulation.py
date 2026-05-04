"""
Single-point, geometry optimisation, and molecular dynamics runners.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Literal

import numpy as np
from ase import Atoms
try:
    from ase.filters import ExpCellFilter
except ImportError:
    from ase.constraints import ExpCellFilter  # ASE < 3.23
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.optimize import BFGS, FIRE, LBFGS
from ase.units import fs as _FS

OptimizerName = Literal["FIRE", "BFGS", "LBFGS"]
MDEnsemble = Literal["NVT-Langevin", "NVE-VelocityVerlet"]

# GPa → eV/Å³  (1 GPa = 6.2415091e-3 eV/Å³)
_GPA_TO_EV_ANG3: float = 6.2415091e-3

_OPTIMIZERS: dict[str, type] = {"FIRE": FIRE, "BFGS": BFGS, "LBFGS": LBFGS}


# ---------------------------------------------------------------------------
# Single point
# ---------------------------------------------------------------------------


def run_single_point(atoms: Atoms, calculator: Any) -> dict[str, Any]:
    """Compute single-point energy and forces.

    The input ``atoms`` object is deep-copied and never modified.

    Parameters
    ----------
    atoms:
        ASE Atoms object.
    calculator:
        MACE calculator instance.

    Returns
    -------
    dict
        Keys: ``energy`` (eV), ``forces`` (list[list[float]] — eV/Å),
        ``max_force`` (eV/Å).
    """
    atoms = copy.deepcopy(atoms)
    atoms.calc = calculator

    energy = float(atoms.get_potential_energy())
    forces = atoms.get_forces()
    max_force = float(np.max(np.linalg.norm(forces, axis=1)))

    return {
        "energy": energy,
        "forces": forces.tolist(),
        "max_force": max_force,
    }


# ---------------------------------------------------------------------------
# Geometry optimisation
# ---------------------------------------------------------------------------


def run_optimization(
    atoms: Atoms,
    calculator: Any,
    optimizer: OptimizerName = "FIRE",
    fmax: float = 0.05,
    max_steps: int = 500,
    relax_cell: bool = False,
    pressure_gpa: float = 0.0,
    step_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[Atoms, list[dict[str, Any]]]:
    """Run geometry optimisation and collect per-step convergence data.

    Parameters
    ----------
    atoms:
        ASE Atoms object (deep-copied internally, never modified).
    calculator:
        MACE calculator instance.
    optimizer:
        Optimiser algorithm: ``'FIRE'``, ``'BFGS'``, or ``'LBFGS'``.
    fmax:
        Force convergence threshold (eV/Å).
    max_steps:
        Maximum number of optimiser steps.
    relax_cell:
        Whether to also relax the unit cell vectors.
    pressure_gpa:
        External pressure in GPa (only used when *relax_cell* is True).
    step_callback:
        Optional callable invoked after each step with the step-data dict.

    Returns
    -------
    tuple[Atoms, list[dict]]
        Relaxed Atoms object and a list of per-step data dictionaries
        (keys: ``step``, ``energy``, ``max_force``).
    """
    atoms = copy.deepcopy(atoms)
    atoms.calc = calculator

    history: list[dict[str, Any]] = []

    system: Any = atoms
    if relax_cell:
        pressure_ev_ang3 = pressure_gpa * _GPA_TO_EV_ANG3
        system = ExpCellFilter(atoms, scalar_pressure=pressure_ev_ang3)

    opt_cls = _OPTIMIZERS[optimizer]
    opt = opt_cls(system, logfile=None)

    def _record() -> None:
        energy = float(atoms.get_potential_energy())
        forces = atoms.get_forces()
        max_force = float(np.max(np.linalg.norm(forces, axis=1)))
        entry: dict[str, Any] = {
            "step": len(history),
            "energy": energy,
            "max_force": max_force,
        }
        history.append(entry)
        if step_callback is not None:
            step_callback(entry)

    # Record the initial state before any steps
    _record()
    opt.attach(_record)
    opt.run(fmax=fmax, steps=max_steps)

    return atoms, history


# ---------------------------------------------------------------------------
# Molecular dynamics
# ---------------------------------------------------------------------------


def run_md(
    atoms: Atoms,
    calculator: Any,
    ensemble: MDEnsemble = "NVT-Langevin",
    temperature_k: float = 300.0,
    timestep_fs: float = 1.0,
    n_steps: int = 1000,
    friction: float = 0.01,
    step_callback: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Run molecular dynamics and collect trajectory data.

    Parameters
    ----------
    atoms:
        ASE Atoms object (deep-copied internally, never modified).
    calculator:
        MACE calculator instance.
    ensemble:
        ``'NVT-Langevin'`` or ``'NVE-VelocityVerlet'``.
    temperature_k:
        Target temperature in Kelvin (NVT only).
    timestep_fs:
        MD timestep in femtoseconds.
    n_steps:
        Number of MD steps to run.
    friction:
        Langevin friction coefficient in fs⁻¹ (NVT only).
    step_callback:
        Optional callable invoked after each step with the step-data dict.

    Returns
    -------
    list[dict]
        Per-step data: ``step``, ``time_fs``, ``energy``, ``temperature_k``.
    """
    atoms = copy.deepcopy(atoms)
    atoms.calc = calculator

    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_k)

    history: list[dict[str, Any]] = []
    dt = timestep_fs * _FS

    if ensemble == "NVT-Langevin":
        dyn: Any = Langevin(
            atoms,
            timestep=dt,
            temperature_K=temperature_k,
            friction=friction / _FS,
            logfile=None,
        )
    else:
        dyn = VelocityVerlet(atoms, timestep=dt, logfile=None)

    def _record() -> None:
        step = len(history)
        entry: dict[str, Any] = {
            "step": step,
            "time_fs": step * timestep_fs,
            "energy": float(atoms.get_potential_energy()),
            "temperature_k": float(atoms.get_temperature()),
        }
        history.append(entry)
        if step_callback is not None:
            step_callback(entry)

    # Record the initial state (t = 0)
    _record()
    dyn.attach(_record, interval=1)
    dyn.run(n_steps)

    return history
