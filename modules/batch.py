"""
Batch "Generate → Relax → Rank" workflow engine.
"""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd
from ase import Atoms

from .simulation import OptimizerName, run_optimization


def batch_relax_and_rank(
    structures: list[Atoms],
    labels: list[str],
    calculator: Any,
    optimizer: OptimizerName = "FIRE",
    fmax_coarse: float = 0.1,
    fmax_fine: float = 0.01,
    max_steps_coarse: int = 300,
    max_steps_fine: int = 500,
    relax_cell: bool = True,
    pressure_gpa: float = 0.0,
    progress_callback: Callable[[int, int, str, str], None] | None = None,
) -> pd.DataFrame:
    """Batch-relax structures with a two-step protocol and rank by energy.

    The two-step protocol mirrors typical DFT/ML screening workflows:

    1. **Coarse relaxation** (``fmax_coarse``) — fast rough geometry
       optimisation to remove obvious bad contacts.
    2. **Fine relaxation** (``fmax_fine``) — tight-convergence run starting
       from the coarse-relaxed geometry.

    Structures are ranked by total energy; relative energies
    ``E_i − E_min`` are also computed.

    Parameters
    ----------
    structures:
        List of ASE Atoms objects to process.
    labels:
        Human-readable label for each structure (same length as *structures*).
    calculator:
        MACE calculator instance.
    optimizer:
        Optimiser algorithm name.
    fmax_coarse, fmax_fine:
        Force convergence thresholds (eV/Å) for each stage.
    max_steps_coarse, max_steps_fine:
        Maximum step counts for each stage.
    relax_cell:
        Whether to allow cell relaxation.
    pressure_gpa:
        External pressure in GPa.
    progress_callback:
        Optional ``callback(i, n_total, label, stage)`` called before each
        stage of each structure.

    Returns
    -------
    pd.DataFrame
        Results table ranked by energy (1-based ``rank`` index) with columns:
        ``label``, ``formula``, ``n_atoms``, ``energy_ev``,
        ``energy_per_atom``, ``e_rel_ev``, ``converged``.
    """
    rows: list[dict[str, Any]] = []
    n = len(structures)

    for i, (atoms, label) in enumerate(zip(structures, labels)):
        # ── Stage 1: coarse ────────────────────────────────────────────────
        if progress_callback is not None:
            progress_callback(i, n, label, "coarse")

        relaxed, history_coarse = run_optimization(
            atoms,
            calculator,
            optimizer=optimizer,
            fmax=fmax_coarse,
            max_steps=max_steps_coarse,
            relax_cell=relax_cell,
            pressure_gpa=pressure_gpa,
        )

        # ── Stage 2: fine ──────────────────────────────────────────────────
        if progress_callback is not None:
            progress_callback(i, n, label, "fine")

        relaxed, history_fine = run_optimization(
            relaxed,
            calculator,
            optimizer=optimizer,
            fmax=fmax_fine,
            max_steps=max_steps_fine,
            relax_cell=relax_cell,
            pressure_gpa=pressure_gpa,
        )

        last = history_fine[-1] if history_fine else history_coarse[-1]
        final_energy: float = last["energy"]
        converged: bool = last["max_force"] <= fmax_fine

        rows.append(
            {
                "label": label,
                "formula": relaxed.get_chemical_formula(),
                "n_atoms": len(relaxed),
                "energy_ev": final_energy,
                "energy_per_atom": final_energy / len(relaxed),
                "converged": converged,
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    e_min: float = df["energy_ev"].min()
    df["e_rel_ev"] = df["energy_ev"] - e_min
    df = df.sort_values("energy_ev").reset_index(drop=True)
    df.index = df.index + 1
    df.index.name = "rank"

    return df
