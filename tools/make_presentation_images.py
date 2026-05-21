"""Create production-ready PNGs and HTMLs for the presentation.

The script renders the same case through four stages on identical axes:
1. Input OBJ as a plain grey mesh
2. Segmentation TXT as a colored point cloud
3. Reconstruction OBJ as a mesh colored from TXT labels
4. Colored reconstruction OBJ with a straight miniplate candidate overlaid

It intentionally does not modify the Streamlit app or visualization helpers.
The reconstruction mesh coloring reuses
``interference_app.visualizations.transfer_txt_colors_to_mesh_vertices``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from interference_app.visualizations import (  # noqa: E402
    COLOR_MAP,
    UNKNOWN_COLOR,
    Mesh,
    _hex_to_rgb,
    combined_figure_colored_bone_mesh_txt,
    load_obj_mesh,
    load_point_cloud_json,
    mesh_figure,
    mesh_figure_colored_by_txt,
    point_cloud_figure,
    save_figure_html,
    transfer_txt_colors_to_mesh_vertices,
)


VIEW_PRESETS: dict[str, tuple[float, float]] = {
    "iso": (18, -58),
    "front": (2, -90),
    "left": (4, 180),
    "right": (4, 0),
}

# Darker than neutral bone-grey (PNG uses RGBA; HTML uses Plotly ``mesh_figure(..., color=...)``).
_INPUT_MESH_RGBA = (0.38, 0.37, 0.36, 0.98)
_INPUT_MESH_HEX = "#5c5a56"


def _rgb01(hex_color: str) -> tuple[float, float, float]:
    r, g, b = _hex_to_rgb(hex_color)
    return r / 255.0, g / 255.0, b / 255.0


def _subsample_faces(mesh: Mesh, max_faces: int, seed: int) -> np.ndarray:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if max_faces <= 0 or len(faces) <= max_faces:
        return faces
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(faces), int(max_faces), replace=False))
    return faces[idx]


def _global_limits(arrays: Iterable[np.ndarray], pad_frac: float = 0.08) -> tuple[tuple[float, float], ...]:
    pts = [np.asarray(a, dtype=np.float64).reshape(-1, 3) for a in arrays if np.asarray(a).size]
    if not pts:
        return ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
    all_pts = np.vstack(pts)
    lo = np.nanmin(all_pts, axis=0)
    hi = np.nanmax(all_pts, axis=0)
    center = (lo + hi) / 2.0
    radius = float(np.max(hi - lo)) / 2.0
    radius = max(radius * (1.0 + pad_frac), 1.0)
    return tuple((float(c - radius), float(c + radius)) for c in center)


def _style_axes(ax, limits: tuple[tuple[float, float], ...], title: str, elev: float, azim: float) -> None:
    ax.set_title(title, fontsize=22, pad=22, fontweight="semibold")
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_zlim(*limits[2])
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_facecolor("#ffffff")
    ax.figure.patch.set_facecolor("#ffffff")


def _mesh_collection(
    mesh: Mesh,
    faces: np.ndarray,
    *,
    facecolors,
    edgecolor=(0.75, 0.75, 0.75, 0.08),
    linewidth: float = 0.02,
    alpha: float = 1.0,
) -> Poly3DCollection:
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    polys = verts[np.asarray(faces, dtype=np.int64)]
    coll = Poly3DCollection(
        polys,
        facecolors=facecolors,
        edgecolors=edgecolor,
        linewidths=linewidth,
        alpha=alpha,
        antialiased=True,
    )
    return coll


def _save(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.08, facecolor=fig.get_facecolor())
    plt.close(fig)


def _render_input_mesh(mesh: Mesh, limits, out_path: Path, view) -> None:
    fig = plt.figure(figsize=(13, 9), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    faces = _subsample_faces(mesh, max_faces=70_000, seed=11)
    coll = _mesh_collection(
        mesh,
        faces,
        facecolors=_INPUT_MESH_RGBA,
        edgecolor=(0.22, 0.22, 0.22, 0.045),
        linewidth=0.015,
    )
    ax.add_collection3d(coll)
    _style_axes(ax, limits, "Input bone mesh", *view)
    _save(fig, out_path)


def _render_segmentation(pc, limits, out_path: Path, view) -> None:
    fig = plt.figure(figsize=(13, 9), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    colors = np.asarray([_rgb01(COLOR_MAP.get(str(label), UNKNOWN_COLOR)) for label in pc.labels])
    pts = np.asarray(pc.xyz, dtype=np.float64)
    if len(pts) > 120_000:
        rng = np.random.default_rng(7)
        idx = np.sort(rng.choice(len(pts), 120_000, replace=False))
        pts = pts[idx]
        colors = colors[idx]
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors, s=0.7, alpha=0.9, linewidths=0)
    _style_axes(ax, limits, "Segmentation labels", *view)
    _save(fig, out_path)


def _render_colored_reconstruction(mesh: Mesh, vertex_rgb: np.ndarray, limits, out_path: Path, view) -> None:
    fig = plt.figure(figsize=(13, 9), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    faces = _subsample_faces(mesh, max_faces=85_000, seed=19)
    face_rgb = np.mean(vertex_rgb[faces].astype(np.float64), axis=1) / 255.0
    face_rgba = np.column_stack([face_rgb, np.full(len(face_rgb), 0.98)])
    coll = _mesh_collection(mesh, faces, facecolors=face_rgba, edgecolor=(0.2, 0.2, 0.2, 0.025))
    ax.add_collection3d(coll)
    _style_axes(ax, limits, "Reconstruction mesh colored by labels", *view)
    _save(fig, out_path)


def _render_plate_overlay(
    bone_mesh: Mesh,
    bone_vertex_rgb: np.ndarray,
    plate_mesh: Mesh,
    limits,
    out_path: Path,
    view,
) -> None:
    fig = plt.figure(figsize=(13, 9), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    bone_faces = _subsample_faces(bone_mesh, max_faces=65_000, seed=23)
    face_rgb = np.mean(bone_vertex_rgb[bone_faces].astype(np.float64), axis=1) / 255.0
    face_rgba = np.column_stack([face_rgb, np.full(len(face_rgb), 0.62)])
    ax.add_collection3d(
        _mesh_collection(
            bone_mesh,
            bone_faces,
            facecolors=face_rgba,
            edgecolor=(0.2, 0.2, 0.2, 0.018),
        )
    )

    plate_faces = _subsample_faces(plate_mesh, max_faces=35_000, seed=29)
    ax.add_collection3d(
        _mesh_collection(
            plate_mesh,
            plate_faces,
            facecolors=(0.10, 0.45, 0.95, 1.0),
            edgecolor=(0.02, 0.08, 0.2, 0.18),
            linewidth=0.06,
        )
    )
    _style_axes(ax, limits, "Colored reconstruction with straight miniplate", *view)
    _save(fig, out_path)


def _fix_plotly_axes(fig, limits: tuple[tuple[float, float], ...]) -> None:
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=list(limits[0]), visible=False),
            yaxis=dict(range=list(limits[1]), visible=False),
            zaxis=dict(range=list(limits[2]), visible=False),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.35, y=1.35, z=1.15)),
        ),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
    )


def _write_htmls(
    *,
    out_dir: Path,
    limits: tuple[tuple[float, float], ...],
    input_mesh: Mesh,
    seg_pc,
    recon_mesh: Mesh,
    recon_pc,
    recon_rgb: np.ndarray,
    plate_mesh: Mesh,
    plate_path: Path,
) -> list[Path]:
    html_dir = out_dir / "html"
    html_dir.mkdir(parents=True, exist_ok=True)

    figs = {
        "BONE_03_EDIT_01_01_input_mesh.html": mesh_figure(
            input_mesh,
            title="Input bone mesh",
            mesh_path=None,
            color=_INPUT_MESH_HEX,
        ),
        "BONE_03_EDIT_01_02_segmentation.html": point_cloud_figure(
            seg_pc,
            title="Segmentation labels",
        ),
        "BONE_03_EDIT_01_03_reconstruction_colored.html": mesh_figure_colored_by_txt(
            recon_mesh,
            recon_pc,
            title="Reconstruction mesh colored by labels",
            mesh_path=None,
            mode="nearest",
            k_smooth=12,
            grey_vertices_not_in_txt_ids=True,
            max_faces=90_000,
        ),
        "BONE_03_EDIT_01_04_miniplate_overlay_colored.html": combined_figure_colored_bone_mesh_txt(
            bone_mesh=recon_mesh,
            bone_vertex_rgb=recon_rgb,
            plate_meshes={1: plate_mesh},
            selected_ids=[1],
            title="Colored reconstruction with straight miniplate",
            bone_path=None,
            bone_opacity=0.72,
            plate_opacity=1.0,
            highlight_ids=[1],
            max_bone_faces=90_000,
        ),
    }
    written: list[Path] = []
    for filename, fig in figs.items():
        _fix_plotly_axes(fig, limits)
        if "miniplate" in filename:
            fig.update_layout(title=f"Colored reconstruction with straight miniplate ({plate_path.name})")
        out = html_dir / filename
        save_figure_html(fig, out)
        written.append(out)
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    ap.add_argument("--out-dir", type=Path, default=Path("presentation_images"))
    ap.add_argument("--views", type=str, default="iso,front,left,right")
    ap.add_argument("--no-html", action="store_true", help="Skip interactive HTML exports.")
    args = ap.parse_args(argv)

    repo_root = args.repo_root.resolve()
    out_dir = (repo_root / args.out_dir).resolve()

    input_obj = repo_root / "interference_app/input_data/BONE_03_EDIT_01.obj"
    segmentation_txt = repo_root / "interference_app/output_data/segmentation/BONE_03_EDIT_01/BONE_03_EDIT_01.txt"
    reconstruction_obj = repo_root / "interference_app/output_data/reconstruction/BONE_03_EDIT_01/BONE_03_EDIT_01.obj"
    reconstruction_txt = repo_root / "interference_app/output_data/reconstruction/BONE_03_EDIT_01/BONE_03_EDIT_01.txt"
    plate_obj = repo_root / "interference_app/output_data/miniplate_positions/BONE_03_EDIT_01/meshes/candidate_0001_original_straight.obj"

    input_mesh = load_obj_mesh(input_obj)
    seg_pc = load_point_cloud_json(segmentation_txt, max_points=None)
    recon_mesh = load_obj_mesh(reconstruction_obj)
    recon_pc = load_point_cloud_json(reconstruction_txt, max_points=None)
    plate_mesh = load_obj_mesh(plate_obj)
    recon_rgb = transfer_txt_colors_to_mesh_vertices(
        recon_mesh,
        recon_pc,
        mode="nearest",
        k_smooth=12,
        grey_vertices_not_in_txt_ids=True,
    )

    limits = _global_limits(
        [
            input_mesh.vertices,
            seg_pc.xyz,
            recon_mesh.vertices,
            plate_mesh.vertices,
        ]
    )

    written: list[Path] = []
    for view_name in [v.strip() for v in args.views.split(",") if v.strip()]:
        if view_name not in VIEW_PRESETS:
            continue
        view = VIEW_PRESETS[view_name]
        jobs = [
            ("01_input_mesh", lambda p: _render_input_mesh(input_mesh, limits, p, view)),
            ("02_segmentation", lambda p: _render_segmentation(seg_pc, limits, p, view)),
            ("03_reconstruction_colored", lambda p: _render_colored_reconstruction(recon_mesh, recon_rgb, limits, p, view)),
            ("04_miniplate_overlay", lambda p: _render_plate_overlay(recon_mesh, recon_rgb, plate_mesh, limits, p, view)),
        ]
        for stem, fn in jobs:
            out = out_dir / f"BONE_03_EDIT_01_{stem}_{view_name}.png"
            fn(out)
            written.append(out)

    if not args.no_html:
        written.extend(
            _write_htmls(
                out_dir=out_dir,
                limits=limits,
                input_mesh=input_mesh,
                seg_pc=seg_pc,
                recon_mesh=recon_mesh,
                recon_pc=recon_pc,
                recon_rgb=recon_rgb,
                plate_mesh=plate_mesh,
                plate_path=plate_obj,
            )
        )

    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

