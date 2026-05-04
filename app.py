"""
CMN-MACE Studio — Main Streamlit Application

Research-grade web platform for atomistic simulations using ASE + MACE.
Developed at CMN · UMONS — Machine Learning Potentials for Materials Discovery.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

import config
from modules.batch import batch_relax_and_rank
from modules.calculator import detect_device, get_calculator, is_mace_available
from modules.codegen import generate_script, generate_slurm
from modules.simulation import run_md, run_optimization, run_single_point
from modules.structure import (
    apply_constraints,
    classify_structure,
    get_structure_summary,
    load_all_frames,
    load_structure,
    suggest_parameters,
)
from modules.visualization import (
    plot_batch_ranking,
    plot_energy_vs_step,
    plot_md_trajectory,
)

# ── Page configuration ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title=config.APP_NAME,
    page_icon="logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session-state initialisation ────────────────────────────────────────────────
_DEFAULTS: dict[str, Any] = {
    "atoms": None,
    "structure_filename": None,
    "structure_type": None,
    "sp_results": None,
    "opt_history": None,
    "opt_atoms": None,
    "md_history": None,
    "batch_results": None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sidebar — Model configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.sidebar:
    st.image("logo.png", width="stretch")
    st.caption(f"`v{config.APP_VERSION}` · CMN · UMONS")
    st.divider()

    st.subheader("Model")
    model_choice: str = st.selectbox(
        "Model size",
        config.MACE_MODELS + ["custom"],
        index=1,
        help="Pre-trained MACE-MP universal model size, or supply your own.",
    )  # type: ignore[assignment]

    custom_model_path: str = ""
    if model_choice == "custom":
        custom_model_path = st.text_input(
            "Path to .model file",
            placeholder="/path/to/model.model",
        )

    # Resolve the effective model identifier
    if model_choice == "custom":
        model: str = custom_model_path if custom_model_path else "medium"
    else:
        model = model_choice

    st.subheader("Compute")
    _detected = detect_device()
    device_choice: str = st.selectbox(
        "Device",
        ["auto", "cpu", "cuda"],
        index=0,
        help=f"Detected: **{_detected}**",
    )  # type: ignore[assignment]
    device: str = _detected if device_choice == "auto" else device_choice

    dtype_choice: str = st.selectbox(
        "Precision",
        config.DTYPES,
        index=0,
    )  # type: ignore[assignment]

    st.divider()

    _calc_ready = False
    if not is_mace_available():
        st.error(
            "⚠️ **MACE not available** — `matscipy` could not be imported.\n\n"
            "To fix on Windows, use conda:\n"
            "```\nconda install -c conda-forge mace-torch\n```"
        )
    elif st.button("🔧 Load / Reload Calculator", type="primary", use_container_width=True):
        if model_choice == "custom" and not custom_model_path:
            st.error("Please enter a path to your custom .model file.")
        else:
            try:
                get_calculator(model, device, dtype_choice)  # type: ignore[arg-type]
                _calc_ready = True
                st.success(f"Model **{model}** loaded on **{device}**")
            except Exception as exc:
                st.error(f"Failed to load calculator:\n\n{exc}")

    # Show persistent status indicator
    if not is_mace_available():
        pass  # error already shown above
    else:
        try:
            # Attempt a no-op to check if the resource is cached
            get_calculator(model, device, dtype_choice)  # type: ignore[arg-type]
            st.success("✅ Calculator ready")
        except Exception:
            st.warning("⚠️ Calculator not loaded yet")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main panel
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
col_logo, col_title = st.columns([1, 4])
with col_logo:
    st.image("logo.png", width=180)
with col_title:
    st.title(config.APP_NAME)
    st.caption(
        f"v{config.APP_VERSION} · Machine Learning Potentials for Materials Discovery · "
        "CMN · UMONS · ASE + MACE"
    )

tab_structure, tab_simulate, tab_batch, tab_hpc = st.tabs(
    ["📁 Structure", "⚙️ Simulate", "🔬 Batch Workflow", "💾 HPC Export"]
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tab 1 — Structure
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_structure:
    st.header("Structure Loading")
    st.info(
        "Supported formats: **POSCAR/CONTCAR** (.vasp), **CIF**, **XYZ**, "
        "**Extended XYZ**, **ASE trajectory** (.traj).  "
        "POSCAR files without an extension should be renamed to `structure.vasp`."
    )

    uploaded_file = st.file_uploader(
        "Upload a structure file",
        type=config.SUPPORTED_EXTENSIONS,
        help="Drop or browse for a structure file.",
    )

    if uploaded_file is not None:
        with st.spinner("Reading structure…"):
            try:
                atoms = load_structure(uploaded_file.getvalue(), uploaded_file.name)
                atoms = apply_constraints(atoms)
                struct_type = classify_structure(atoms)
                st.session_state["atoms"] = atoms
                st.session_state["structure_filename"] = uploaded_file.name
                st.session_state["structure_type"] = struct_type
            except Exception as exc:
                st.error(f"Could not read structure: {exc}")

    if st.session_state["atoms"] is not None:
        _atoms = st.session_state["atoms"]
        _stype: str = st.session_state["structure_type"]
        summary = get_structure_summary(_atoms)

        st.success(f"Loaded: **{summary['formula']}** — {summary['n_atoms']} atoms")

        col_info, col_cell = st.columns(2)

        with col_info:
            st.subheader("Summary")
            m1, m2, m3 = st.columns(3)
            m1.metric("Formula", summary["formula"])
            m2.metric("Atoms", summary["n_atoms"])
            m3.metric("Type", _stype.capitalize())
            m4, m5 = st.columns(2)
            m4.metric("Species", ", ".join(summary["species"]))
            m5.metric("PBC", str(summary["pbc"]))
            if summary["has_constraints"]:
                st.info(
                    "🔒 Selective Dynamics detected — "
                    "FixAtoms constraints applied."
                )

        with col_cell:
            st.subheader("Unit Cell (Å)")
            cell_arr = np.array(summary["cell"])
            df_cell = pd.DataFrame(
                cell_arr,
                index=["**a**", "**b**", "**c**"],
                columns=["x", "y", "z"],
            )
            st.dataframe(df_cell.style.format("{:.4f}"), use_container_width=True)

            lengths = np.linalg.norm(cell_arr, axis=1)
            la, lb, lc = st.columns(3)
            la.metric("|a|", f"{lengths[0]:.4f} Å")
            lb.metric("|b|", f"{lengths[1]:.4f} Å")
            lc.metric("|c|", f"{lengths[2]:.4f} Å")

        # Smart parameter suggestions
        st.subheader("💡 Smart Parameter Suggestions")
        suggestions = suggest_parameters(_stype)
        s_cols = st.columns(len(suggestions))
        for col, (key, val) in zip(s_cols, suggestions.items()):
            col.metric(key.replace("_", " ").title(), val)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tab 2 — Simulate
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_simulate:
    st.header("Simulation")

    if st.session_state["atoms"] is None:
        st.warning("Upload a structure in the **Structure** tab first.")
    else:
        _atoms = st.session_state["atoms"]
        _stype = st.session_state["structure_type"]
        suggestions = suggest_parameters(_stype)

        sim_mode: str = st.radio(
            "Simulation Mode",
            ["Single Point", "Geometry Optimisation", "Molecular Dynamics"],
            horizontal=True,
        )  # type: ignore[assignment]

        st.divider()

        # ────────────────────────────────────────────────────────────────────
        # Single Point
        # ────────────────────────────────────────────────────────────────────
        if sim_mode == "Single Point":
            st.subheader("Single Point Calculation")

            if st.button("▶ Run Single Point", type="primary"):
                calc = get_calculator(model, device, dtype_choice)  # type: ignore[arg-type]
                with st.spinner("Computing energy and forces…"):
                    try:
                        results = run_single_point(_atoms, calc)
                        st.session_state["sp_results"] = results
                    except Exception as exc:
                        st.error(f"Calculation failed: {exc}")

            if st.session_state["sp_results"] is not None:
                res = st.session_state["sp_results"]
                st.success("Calculation complete!")
                c1, c2, c3 = st.columns(3)
                c1.metric("Energy", f"{res['energy']:.6f} eV")
                c2.metric("Max Force", f"{res['max_force']:.4f} eV/Å")
                c3.metric("Atoms", len(_atoms))

                with st.expander("View Forces"):
                    forces_df = pd.DataFrame(
                        res["forces"],
                        columns=["Fx (eV/Å)", "Fy (eV/Å)", "Fz (eV/Å)"],
                    )
                    forces_df.index.name = "Atom"
                    st.dataframe(
                        forces_df.style.format("{:.6f}"),
                        use_container_width=True,
                    )

        # ────────────────────────────────────────────────────────────────────
        # Geometry Optimisation
        # ────────────────────────────────────────────────────────────────────
        elif sim_mode == "Geometry Optimisation":
            st.subheader("Geometry Optimisation")

            col_opt1, col_opt2 = st.columns(2)
            with col_opt1:
                optimizer: str = st.selectbox(  # type: ignore[assignment]
                    "Optimiser",
                    config.OPTIMIZERS,
                    index=0,
                    help="FIRE is generally recommended for periodic systems.",
                )
                fmax: float = st.number_input(
                    "fmax (eV/Å)",
                    value=float(suggestions["fmax_fine"]),
                    step=0.005,
                    format="%.4f",
                    min_value=0.001,
                    help="Convergence threshold on the maximum atomic force.",
                )
                max_steps: int = st.number_input(
                    "Max steps",
                    value=500,
                    step=50,
                    min_value=10,
                )  # type: ignore[assignment]

            with col_opt2:
                relax_cell: bool = st.checkbox(
                    "Relax unit cell",
                    value=bool(suggestions["relax_cell"]),
                    help="Use ExpCellFilter to relax cell vectors alongside positions.",
                )
                pressure_gpa: float = st.number_input(
                    "External pressure (GPa)",
                    value=float(suggestions["pressure_gpa"]),
                    step=0.1,
                    format="%.2f",
                    disabled=not relax_cell,
                )

            if st.button("▶ Run Optimisation", type="primary"):
                calc = get_calculator(model, device, dtype_choice)  # type: ignore[arg-type]
                progress_bar = st.progress(0, text="Optimising…")

                with st.spinner("Running geometry optimisation…"):
                    try:
                        opt_atoms, history = run_optimization(
                            _atoms,
                            calc,
                            optimizer=optimizer,  # type: ignore[arg-type]
                            fmax=fmax,
                            max_steps=max_steps,
                            relax_cell=relax_cell,
                            pressure_gpa=pressure_gpa,
                        )
                        st.session_state["opt_history"] = history
                        st.session_state["opt_atoms"] = opt_atoms
                        progress_bar.progress(100, text="Done!")
                    except Exception as exc:
                        progress_bar.empty()
                        st.error(f"Optimisation failed: {exc}")

            if st.session_state["opt_history"] is not None:
                history = st.session_state["opt_history"]
                final = history[-1]
                st.success(f"Optimisation complete — {len(history)} steps")

                r1, r2, r3 = st.columns(3)
                r1.metric("Final Energy", f"{final['energy']:.6f} eV")
                r2.metric("Final Max Force", f"{final['max_force']:.4f} eV/Å")
                r3.metric("Steps", len(history))

                st.plotly_chart(
                    plot_energy_vs_step(history), use_container_width=True
                )

                csv_bytes = pd.DataFrame(history).to_csv(index=False).encode()
                st.download_button(
                    "⬇ Download convergence.csv",
                    data=csv_bytes,
                    file_name="convergence.csv",
                    mime="text/csv",
                )

        # ────────────────────────────────────────────────────────────────────
        # Molecular Dynamics
        # ────────────────────────────────────────────────────────────────────
        elif sim_mode == "Molecular Dynamics":
            st.subheader("Molecular Dynamics")

            col_md1, col_md2 = st.columns(2)
            with col_md1:
                ensemble: str = st.selectbox(  # type: ignore[assignment]
                    "Ensemble",
                    config.MD_ENSEMBLES,
                    index=0,
                )
                temperature_k: float = st.number_input(
                    "Temperature (K)",
                    value=300.0,
                    step=50.0,
                    min_value=1.0,
                )
                timestep_fs: float = st.number_input(
                    "Timestep (fs)",
                    value=1.0,
                    step=0.5,
                    min_value=0.1,
                    format="%.2f",
                )

            with col_md2:
                n_steps: int = st.number_input(
                    "Number of steps",
                    value=500,
                    step=100,
                    min_value=10,
                )  # type: ignore[assignment]
                friction: float = st.number_input(
                    "Friction (fs⁻¹)",
                    value=0.01,
                    step=0.005,
                    format="%.4f",
                    disabled=(ensemble != "NVT-Langevin"),
                    help="Langevin friction coefficient. Only used for NVT.",
                )

            if st.button("▶ Run MD", type="primary"):
                calc = get_calculator(model, device, dtype_choice)  # type: ignore[arg-type]
                with st.spinner(f"Running {ensemble} MD…"):
                    try:
                        md_history = run_md(
                            _atoms,
                            calc,
                            ensemble=ensemble,  # type: ignore[arg-type]
                            temperature_k=temperature_k,
                            timestep_fs=timestep_fs,
                            n_steps=n_steps,
                            friction=friction,
                        )
                        st.session_state["md_history"] = md_history
                    except Exception as exc:
                        st.error(f"MD failed: {exc}")

            if st.session_state["md_history"] is not None:
                md_history = st.session_state["md_history"]
                final_md = md_history[-1]
                st.success(f"MD complete — {len(md_history)} frames")

                m1, m2, m3 = st.columns(3)
                m1.metric("Final Energy", f"{final_md['energy']:.4f} eV")
                m2.metric("Final Temperature", f"{final_md['temperature_k']:.1f} K")
                m3.metric("Total Time", f"{final_md['time_fs']:.1f} fs")

                st.plotly_chart(
                    plot_md_trajectory(md_history), use_container_width=True
                )

                csv_bytes = pd.DataFrame(md_history).to_csv(index=False).encode()
                st.download_button(
                    "⬇ Download md_trajectory.csv",
                    data=csv_bytes,
                    file_name="md_trajectory.csv",
                    mime="text/csv",
                )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tab 3 — Batch Workflow
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_batch:
    st.header("Batch Workflow: Generate → Relax → Rank")
    st.markdown(
        """
        Upload multiple structure files **or** a single `.traj` file (all frames are read).

        Pipeline:
        1. Load all structures  
        2. **Two-step relaxation**: coarse (fast) → fine (tight)  
        3. **Rank by total energy** and report relative energies $E_i - E_{\\min}$
        """
    )

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        batch_files = st.file_uploader(
            "Upload structures (multiple allowed)",
            type=config.SUPPORTED_EXTENSIONS,
            accept_multiple_files=True,
            key="batch_uploader",
        )
        relax_cell_batch: bool = st.checkbox(
            "Relax unit cell", value=True, key="batch_cell"
        )
        pressure_batch: float = st.number_input(
            "Pressure (GPa)",
            value=0.0,
            step=0.1,
            format="%.2f",
            key="batch_pressure",
        )

    with col_b2:
        optimizer_batch: str = st.selectbox(  # type: ignore[assignment]
            "Optimiser",
            config.OPTIMIZERS,
            key="batch_optimizer",
        )
        fmax_coarse: float = st.number_input(
            "fmax coarse (eV/Å)",
            value=0.1,
            step=0.01,
            format="%.3f",
            min_value=0.001,
        )
        fmax_fine: float = st.number_input(
            "fmax fine (eV/Å)",
            value=0.01,
            step=0.005,
            format="%.4f",
            min_value=0.001,
        )

    if batch_files and st.button("▶ Run Batch Relax & Rank", type="primary"):
        calc = get_calculator(model, device, dtype_choice)  # type: ignore[arg-type]
        structures: list = []
        labels: list[str] = []

        progress_bar = st.progress(0, text="Loading structures…")

        for i, bfile in enumerate(batch_files):
            progress_bar.progress(
                int(100 * i / len(batch_files)),
                text=f"Loading {bfile.name}…",
            )
            suffix = Path(bfile.name).suffix.lower()
            stem = Path(bfile.name).stem

            if suffix == ".traj":
                try:
                    frames = load_all_frames(bfile.getvalue(), bfile.name)
                    for j, frame in enumerate(frames):
                        structures.append(apply_constraints(frame))
                        labels.append(f"{stem}_f{j:03d}")
                except Exception as exc:
                    st.warning(f"Could not load {bfile.name}: {exc}")
            else:
                try:
                    at = load_structure(bfile.getvalue(), bfile.name)
                    structures.append(apply_constraints(at))
                    labels.append(stem)
                except Exception as exc:
                    st.warning(f"Could not load {bfile.name}: {exc}")

        if not structures:
            st.error("No valid structures were loaded.")
        else:
            st.info(f"Loaded {len(structures)} structure(s). Starting relaxation…")
            _prog_bar = st.progress(0, text="Relaxing…")

            def _progress_cb(i: int, n: int, label: str, stage: str) -> None:
                _prog_bar.progress(
                    int(100 * i / n),
                    text=f"[{i + 1}/{n}] {label} — {stage} relaxation…",
                )

            try:
                df_batch = batch_relax_and_rank(
                    structures,
                    labels,
                    calc,
                    optimizer=optimizer_batch,  # type: ignore[arg-type]
                    fmax_coarse=fmax_coarse,
                    fmax_fine=fmax_fine,
                    relax_cell=relax_cell_batch,
                    pressure_gpa=pressure_batch,
                    progress_callback=_progress_cb,
                )
                st.session_state["batch_results"] = df_batch
                _prog_bar.progress(100, text="Done!")
            except Exception as exc:
                _prog_bar.empty()
                st.error(f"Batch workflow failed: {exc}")

    if st.session_state["batch_results"] is not None:
        df_batch = st.session_state["batch_results"]
        st.success(f"Batch complete — {len(df_batch)} structures ranked.")

        fmt_cols = {
            "energy_ev": "{:.6f}",
            "energy_per_atom": "{:.6f}",
            "e_rel_ev": "{:.6f}",
        }
        st.dataframe(
            df_batch.style.format(fmt_cols),
            use_container_width=True,
        )

        st.plotly_chart(plot_batch_ranking(df_batch), use_container_width=True)

        csv_bytes = df_batch.to_csv().encode()
        st.download_button(
            "⬇ Download batch_ranking.csv",
            data=csv_bytes,
            file_name="batch_ranking.csv",
            mime="text/csv",
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tab 4 — HPC Export
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_hpc:
    st.header("HPC Export")
    st.markdown(
        "Generate standalone Python scripts and SLURM submission scripts "
        "ready to be copied to an HPC cluster."
    )

    if st.session_state["atoms"] is None:
        st.warning("Upload a structure in the **Structure** tab first.")
    else:
        tab_py, tab_slurm = st.tabs(["🐍 run_mace.py", "🖥 submit.slurm"])

        # ────────────────────────────────────────────────────────────────────
        # Python script
        # ────────────────────────────────────────────────────────────────────
        with tab_py:
            st.subheader("Script Generator")

            c_py1, c_py2 = st.columns(2)
            with c_py1:
                script_mode: str = st.selectbox(  # type: ignore[assignment]
                    "Simulation mode",
                    ["single_point", "optimization", "md"],
                    format_func=lambda x: x.replace("_", " ").title(),
                    key="script_mode",
                )
                script_device: str = st.selectbox(  # type: ignore[assignment]
                    "Device",
                    ["auto", "cpu", "cuda"],
                    key="script_device",
                )
                script_dtype: str = st.selectbox(  # type: ignore[assignment]
                    "Precision",
                    config.DTYPES,
                    key="script_dtype",
                )
            with c_py2:
                script_model: str = st.selectbox(  # type: ignore[assignment]
                    "Model",
                    config.MACE_MODELS,
                    index=1,
                    key="script_model",
                )
                script_filename: str = st.text_input(
                    "Structure filename (in the HPC directory)",
                    value=st.session_state["structure_filename"] or "structure.vasp",
                    key="script_fname",
                )

            # Mode-specific parameters
            _script_kwargs: dict[str, Any] = {}

            if script_mode == "optimization":
                st.subheader("Optimisation parameters")
                op1, op2, op3 = st.columns(3)
                _script_kwargs["optimizer"] = op1.selectbox(
                    "Optimiser", config.OPTIMIZERS, key="s_opt"
                )
                _script_kwargs["fmax"] = op2.number_input(
                    "fmax", value=0.05, step=0.01, format="%.4f", key="s_fmax"
                )
                _script_kwargs["max_steps"] = op3.number_input(
                    "Max steps", value=500, key="s_maxsteps"
                )
                _script_kwargs["relax_cell"] = st.checkbox(
                    "Relax cell", key="s_relax_cell"
                )
                _script_kwargs["pressure_gpa"] = st.number_input(
                    "Pressure (GPa)",
                    value=0.0,
                    key="s_pressure",
                    disabled=not _script_kwargs["relax_cell"],
                )

            elif script_mode == "md":
                st.subheader("MD parameters")
                md1, md2, md3 = st.columns(3)
                _script_kwargs["ensemble"] = md1.selectbox(
                    "Ensemble", config.MD_ENSEMBLES, key="s_ensemble"
                )
                _script_kwargs["temperature_k"] = md2.number_input(
                    "Temperature (K)", value=300.0, key="s_temp"
                )
                _script_kwargs["timestep_fs"] = md3.number_input(
                    "Timestep (fs)", value=1.0, format="%.2f", key="s_dt"
                )
                _script_kwargs["n_steps"] = st.number_input(
                    "Steps", value=1000, key="s_nsteps"
                )
                _script_kwargs["friction"] = st.number_input(
                    "Friction (fs⁻¹)",
                    value=0.01,
                    format="%.4f",
                    key="s_friction",
                    disabled=(_script_kwargs.get("ensemble") != "NVT-Langevin"),
                )

            if st.button("📝 Generate Script", type="primary", key="btn_gen_py"):
                script_content = generate_script(
                    structure_filename=script_filename,
                    model=script_model,
                    device=script_device,
                    dtype=script_dtype,
                    mode=script_mode,  # type: ignore[arg-type]
                    **_script_kwargs,
                )
                st.code(script_content, language="python")
                st.download_button(
                    "⬇ Download run_mace.py",
                    data=script_content.encode(),
                    file_name="run_mace.py",
                    mime="text/x-python",
                    key="dl_run_mace",
                )

        # ────────────────────────────────────────────────────────────────────
        # SLURM script
        # ────────────────────────────────────────────────────────────────────
        with tab_slurm:
            st.subheader("SLURM Script Generator")

            sl1, sl2 = st.columns(2)
            with sl1:
                slurm_job_name: str = st.text_input("Job name", value="mace_job")
                slurm_nodes: int = st.number_input("Nodes", value=1, min_value=1)  # type: ignore[assignment]
                slurm_ntasks: int = st.number_input("Tasks", value=1, min_value=1)  # type: ignore[assignment]
                slurm_cpus: int = st.number_input("CPUs per task", value=8, min_value=1)  # type: ignore[assignment]
            with sl2:
                slurm_mem: int = st.number_input("Memory (GB)", value=32, min_value=1)  # type: ignore[assignment]
                slurm_time: int = st.number_input(  # type: ignore[assignment]
                    "Wall time (hours)", value=24, min_value=1
                )
                slurm_partition: str = st.text_input("Partition", value="gpu")
                slurm_gpu: bool = st.checkbox("Request GPU", value=True)
                slurm_env: str = st.text_input("Conda environment", value="mace")

            if st.button("📝 Generate SLURM Script", type="primary", key="btn_gen_slurm"):
                slurm_content = generate_slurm(
                    job_name=slurm_job_name,
                    nodes=slurm_nodes,
                    ntasks=slurm_ntasks,
                    cpus_per_task=slurm_cpus,
                    mem_gb=slurm_mem,
                    time_hours=slurm_time,
                    partition=slurm_partition,
                    use_gpu=slurm_gpu,
                    conda_env=slurm_env,
                )
                st.code(slurm_content, language="bash")
                st.download_button(
                    "⬇ Download submit.slurm",
                    data=slurm_content.encode(),
                    file_name="submit.slurm",
                    mime="text/plain",
                    key="dl_slurm",
                )
