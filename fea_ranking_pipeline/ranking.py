"""Ranking of miniplate candidates by geometric and FEA metrics."""

from __future__ import annotations

import pandas as pd

COLUMN_RENAMES: dict[str, str] = {
    "candidate_id": "Candidate",
    "rmse_mm": "rmse_MM",
    "green_pct": "Safe (%)",
    "yellow_pct": "Caution (%)",
    "orange_pct": "Risk (%)",
    "purple_pct": "Critical (%)",
    "fea_fracture_motion_mm": "Fracture motion (mm)",
    "fea_local_fracture_motion_mm": "Local motion (mm)",
    "fea_interface_pair_mean_mm": "Interface mean (mm)",
    "fea_interface_pair_p95_mm": "Interface Max (mm)",
    "fea_max_displacement_mm": "Max disp (mm)",
    "fea_mean_displacement_mm": "Mean disp (mm)",
    "fea_rms_displacement_mm": "RMS disp (mm)",
    "fea_max_von_mises_MPa": "Max stress (MPa)",
    "fea_mean_von_mises_MPa": "Mean stress (MPa)",
}

FEA_METRIC_COLUMNS: tuple[str, ...] = (
    "Fracture motion (mm)",
    "Local motion (mm)",
    "Interface mean (mm)",
    "Interface Max (mm)",
    "Max disp (mm)",
    "Mean disp (mm)",
    "RMS disp (mm)",
    "Max stress (MPa)",
    "Mean stress (MPa)",
)

_COLOR_SCORE = "Color score (Safe+Caution)"
_RANK_COLOR = "Rank color"
_RANK_RMSE = "Rank rmse"
_AVG_FEA_RANK = "Avg FEA rank"
_FINAL_AVG_RANK = "Final avg rank"
_RANK_OVERALL = "Rank overall"

MERGED_RANKING_VIEW_COLUMNS: tuple[str, ...] = (
    _RANK_OVERALL,
    "Candidate",
    "rmse_MM",
    "Safe (%)",
    *FEA_METRIC_COLUMNS,
)


def subset_merged_ranking_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only merged-view columns (exclude per-metric ``Rank …`` helper columns)."""
    cols = [c for c in MERGED_RANKING_VIEW_COLUMNS if c in df.columns]
    return df[cols].copy()


def compute_rankings(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns, compute per-metric ranks, and derive the final average rank.

    Conventions:
      - ``rmse_MM`` and all FEA metrics are ranked ascending (lower is better).
      - ``Safe (%) + Caution (%)`` is ranked descending (higher is better).
      - ``Avg FEA rank`` is the mean of all individual FEA metric ranks.
      - ``Final avg rank`` is the mean of (``Avg FEA rank``, ``Rank rmse``, ``Rank color``).
      - Rows are sorted ascending by ``Final avg rank``.
    """
    out = df.rename(columns={k: v for k, v in COLUMN_RENAMES.items() if k in df.columns}).copy()

    if "Safe (%)" in out.columns and "Caution (%)" in out.columns:
        out[_COLOR_SCORE] = out["Safe (%)"].astype(float) + out["Caution (%)"].astype(float)
        out[_RANK_COLOR] = out[_COLOR_SCORE].rank(ascending=False, method="min")

    if "rmse_MM" in out.columns:
        out[_RANK_RMSE] = out["rmse_MM"].astype(float).rank(ascending=True, method="min")

    fea_rank_cols: list[str] = []
    for col in FEA_METRIC_COLUMNS:
        if col in out.columns:
            rc = f"Rank {col}"
            out[rc] = out[col].astype(float).rank(ascending=True, method="min")
            fea_rank_cols.append(rc)
    if fea_rank_cols:
        out[_AVG_FEA_RANK] = out[fea_rank_cols].mean(axis=1)

    final_parts = [c for c in (_AVG_FEA_RANK, _RANK_RMSE, _RANK_COLOR) if c in out.columns]
    if final_parts:
        out[_FINAL_AVG_RANK] = out[final_parts].mean(axis=1)
        out[_RANK_OVERALL] = out[_FINAL_AVG_RANK].rank(ascending=True, method="min")
        sort_keys = [_FINAL_AVG_RANK]
        if "Candidate" in out.columns:
            sort_keys.append("Candidate")
        out = out.sort_values(sort_keys, ascending=True).reset_index(drop=True)

    preferred_order = [
        c
        for c in (
            _RANK_OVERALL,
            "Candidate",
            "rmse_MM",
            "Safe (%)",
            "Caution (%)",
            "Risk (%)",
            "Critical (%)",
            _COLOR_SCORE,
            _RANK_COLOR,
            _RANK_RMSE,
            _AVG_FEA_RANK,
            _FINAL_AVG_RANK,
        )
        if c in out.columns
    ]
    remaining = [c for c in out.columns if c not in preferred_order]
    return out[preferred_order + remaining]
