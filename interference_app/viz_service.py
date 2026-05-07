"""Bridge between run artifacts and interactive Plotly figures.

Given a run directory, this module resolves the correct input files from
the manifest, builds the corresponding Plotly figures, and writes stand-
alone HTML snapshots under ``<run_dir>/<stage>/viz/``. The UI uses these
helpers to render inline 3D views and offer HTML downloads.

All heavy-weight loading (multi-hundred-thousand point meshes) happens
once per (run, stage). Downstream callers can wrap the returned figures
with Streamlit caching for fast re-rendering.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import plotly.graph_objects as go

from .visualizations import (
    Mesh,
    PointCloud,
    candidates_overview_figure,
    combined_figure,
    load_obj_mesh,
    load_point_cloud_json,
    mesh_figure,
    point_cloud_figure,
    save_figure_html,
)

_CANDIDATE_ID_RE = re.compile(r"candidate_(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class StageArtifacts:
    segmented_txt: Path | None
    segmented_obj: Path | None
    reconstructed_txt: Path | None
    reconstructed_obj: Path | None
    miniplate_meshes_dir: Path | None
    miniplate_summary_json: Path | None


def _stage_artifacts(manifest: Mapping[str, Any]) -> StageArtifacts:
    stages = manifest.get("stages") or {}

    def _path(stage: str, key: str) -> Path | None:
        info = stages.get(stage) or {}
        val = (info.get("artifacts") or {}).get(key)
        return Path(val) if val else None

    return StageArtifacts(
        segmented_txt=_path("segmentation", "segmented_txt"),
        segmented_obj=_path("segmentation", "segmented_obj"),
        reconstructed_txt=_path("reconstruction", "reconstructed_txt"),
        reconstructed_obj=_path("reconstruction", "reconstructed_obj"),
        miniplate_meshes_dir=_path("miniplate", "meshes_dir"),
        miniplate_summary_json=_path("miniplate", "summary_json"),
    )


def _stage_viz_dir(run_dir: Path, stage: str) -> Path:
    out = Path(run_dir) / stage / "viz"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _maybe_write_html(fig: go.Figure, path: Path) -> Path | None:
    if path.is_file():
        return path
    try:
        return save_figure_html(fig, path)
    except Exception:  # noqa: BLE001
        return None


def build_segmentation_figure(run_dir: Path, manifest: Mapping[str, Any]) -> go.Figure | None:
    a = _stage_artifacts(manifest)
    if a.segmented_txt is None or not a.segmented_txt.is_file():
        return None
    pc = load_point_cloud_json(a.segmented_txt)
    title = f"Segmented point cloud ({pc.xyz.shape[0]} pts shown)"
    fig = point_cloud_figure(pc, title=title)
    _maybe_write_html(fig, _stage_viz_dir(run_dir, "segmentation") / "segmentation.html")
    return fig


def build_reconstruction_figure(
    run_dir: Path, manifest: Mapping[str, Any]
) -> go.Figure | None:
    a = _stage_artifacts(manifest)
    if a.reconstructed_txt is not None and a.reconstructed_txt.is_file():
        pc = load_point_cloud_json(a.reconstructed_txt)
        if pc.xyz.shape[0] > 0:
            title = (
                f"Reconstructed point cloud ({pc.xyz.shape[0]} pts shown)"
            )
            fig = point_cloud_figure(pc, title=title)
            _maybe_write_html(
                fig, _stage_viz_dir(run_dir, "reconstruction") / "reconstruction.html"
            )
            return fig
    if a.reconstructed_obj is None or not a.reconstructed_obj.is_file():
        return None
    mesh = load_obj_mesh(a.reconstructed_obj)
    title = (
        f"Reconstructed bone mesh "
        f"({mesh.vertices.shape[0]} verts, {mesh.faces.shape[0]} faces)"
    )
    fig = mesh_figure(mesh, title=title, mesh_path=a.reconstructed_obj)
    _maybe_write_html(fig, _stage_viz_dir(run_dir, "reconstruction") / "reconstruction.html")
    return fig


def _infer_candidate_id(obj_path: Path) -> int | None:
    m = _CANDIDATE_ID_RE.search(obj_path.name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def list_candidate_meshes(manifest: Mapping[str, Any]) -> dict[int, Path]:
    a = _stage_artifacts(manifest)
    out: dict[int, Path] = {}
    if a.miniplate_meshes_dir is None or not a.miniplate_meshes_dir.is_dir():
        return out
    for p in sorted(a.miniplate_meshes_dir.glob("candidate_*.obj")):
        cid = _infer_candidate_id(p)
        if cid is None:
            continue
        out.setdefault(cid, p)
    return out


def load_candidate_meshes(paths: Mapping[int, Path]) -> dict[int, Mesh]:
    return {cid: load_obj_mesh(p) for cid, p in paths.items()}


def _load_reconstructed_mesh(manifest: Mapping[str, Any]) -> tuple[Mesh | None, Path | None]:
    a = _stage_artifacts(manifest)
    if a.reconstructed_obj is None or not a.reconstructed_obj.is_file():
        return None, None
    return load_obj_mesh(a.reconstructed_obj), a.reconstructed_obj


def build_miniplate_overview_figure(
    run_dir: Path, manifest: Mapping[str, Any]
) -> go.Figure | None:
    bone_mesh, bone_path = _load_reconstructed_mesh(manifest)
    if bone_mesh is None:
        return None
    paths = list_candidate_meshes(manifest)
    plates = load_candidate_meshes(paths)
    if not plates:
        return None
    title = f"Miniplate candidates overview ({len(plates)} plates)"
    fig = candidates_overview_figure(bone_mesh, plates, title=title, bone_path=bone_path)
    _maybe_write_html(fig, _stage_viz_dir(run_dir, "miniplate") / "candidates_overview.html")
    return fig


def build_miniplate_selection_figure(
    manifest: Mapping[str, Any],
    selected_ids: Iterable[int],
    *,
    highlight_ids: Iterable[int] | None = None,
    title: str | None = None,
) -> go.Figure | None:
    bone_mesh, bone_path = _load_reconstructed_mesh(manifest)
    if bone_mesh is None:
        return None
    paths = list_candidate_meshes(manifest)
    if not paths:
        return None
    sel = [int(c) for c in selected_ids if int(c) in paths]
    plates = load_candidate_meshes({cid: paths[cid] for cid in sel})
    fig_title = title or f"Bone + {len(sel)} selected candidate(s)"
    return combined_figure(
        bone_mesh=bone_mesh,
        plate_meshes=plates,
        selected_ids=sel,
        title=fig_title,
        bone_path=bone_path,
        highlight_ids=highlight_ids,
    )


def build_top_n_figure(
    run_dir: Path,
    manifest: Mapping[str, Any],
    top_ids: Iterable[int],
) -> go.Figure | None:
    top = [int(x) for x in top_ids]
    fig = build_miniplate_selection_figure(
        manifest,
        selected_ids=top,
        highlight_ids=top,
        title=f"Top {len(top)} miniplate positions",
    )
    if fig is not None:
        _maybe_write_html(fig, _stage_viz_dir(run_dir, "ranking") / "top_positions.html")
    return fig


def summary_candidate_ids(manifest: Mapping[str, Any]) -> list[int]:
    a = _stage_artifacts(manifest)
    if a.miniplate_summary_json is None or not a.miniplate_summary_json.is_file():
        return []
    try:
        data = json.loads(a.miniplate_summary_json.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []
    cands = data.get("all_candidates") or data.get("candidates") or []
    ids: list[int] = []
    for c in cands:
        cid = c.get("candidate_id") if isinstance(c, dict) else None
        if cid is None:
            continue
        try:
            ids.append(int(cid))
        except (TypeError, ValueError):
            continue
    return ids


__all__ = [
    "StageArtifacts",
    "build_miniplate_overview_figure",
    "build_miniplate_selection_figure",
    "build_reconstruction_figure",
    "build_segmentation_figure",
    "build_top_n_figure",
    "list_candidate_meshes",
    "load_candidate_meshes",
    "summary_candidate_ids",
]
