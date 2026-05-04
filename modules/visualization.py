"""
Plotly visualisation helpers for convergence and MD trajectory data.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_energy_vs_step(history: list[dict[str, Any]]) -> go.Figure:
    """Dual-axis convergence plot: energy and max force vs optimiser step."""
    df = pd.DataFrame(history)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["step"],
            y=df["energy"],
            name="Energy (eV)",
            line=dict(color="#1f77b4", width=2),
            yaxis="y1",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["step"],
            y=df["max_force"],
            name="Max Force (eV/Å)",
            line=dict(color="#ff7f0e", width=2, dash="dot"),
            yaxis="y2",
        )
    )

    fig.update_layout(
        title="Geometry Optimisation Convergence",
        xaxis=dict(title=dict(text="Step")),
        yaxis=dict(
            title=dict(
                text="Energy (eV)", 
                font=dict(color="#1f77b4")
            )
        ),
        yaxis2=dict(
            title=dict(
                text="Max Force (eV/Å)", 
                font=dict(color="#ff7f0e")
            ),
            overlaying="y",
            side="right",
        ),
        legend=dict(x=0.01, y=0.99),
        hovermode="x unified",
        template="plotly_white",
        height=420,
    )
    return fig


def plot_md_trajectory(history: list[dict[str, Any]]) -> go.Figure:
    """Dual-axis MD trajectory plot: energy and temperature vs time."""
    df = pd.DataFrame(history)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["time_fs"],
            y=df["energy"],
            name="Energy (eV)",
            line=dict(color="#2ca02c", width=1.5),
            yaxis="y1",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["time_fs"],
            y=df["temperature_k"],
            name="Temperature (K)",
            line=dict(color="#d62728", width=1.5, dash="dot"),
            yaxis="y2",
        )
    )

    fig.update_layout(
        title="MD Trajectory",
        xaxis=dict(title=dict(text="Time (fs)")),
        yaxis=dict(
            title=dict(
                text="Energy (eV)", 
                font=dict(color="#2ca02c")
            )
        ),
        yaxis2=dict(
            title=dict(
                text="Temperature (K)", 
                font=dict(color="#d62728")
            ),
            overlaying="y",
            side="right",
        ),
        legend=dict(x=0.01, y=0.99),
        hovermode="x unified",
        template="plotly_white",
        height=420,
    )
    return fig


def plot_batch_ranking(df: pd.DataFrame) -> go.Figure:
    """Bar chart of relative energies from a batch-ranking result."""
    plot_df = df.reset_index()  # bring 'rank' into columns

    fig = px.bar(
        plot_df,
        x="label",
        y="e_rel_ev",
        color="e_rel_ev",
        color_continuous_scale="Blues_r",
        labels={
            "e_rel_ev": "Relative Energy (eV)",
            "label": "Structure",
            "rank": "Rank",
        },
        hover_data=["rank", "formula", "n_atoms", "energy_ev", "converged"],
        title="Batch Ranking — Relative Energies (E − E_min)",
        template="plotly_white",
    )
    fig.update_layout(coloraxis_showscale=False, height=420)
    return fig
