"""PyVista + stpyvista viewers for smoother mesh shading in Streamlit."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .visualizations import (
    Mesh,
    _seed_from_path,
    _subsample_mesh_faces,
)

PYVISTA_MAX_BONE_FACES: int | None = None
PYVISTA_MAX_PLATE_FACES: int | None = None


def pyvista_stack_available() -> bool:
    try:
        import pyvista as pv  # noqa: F401
        from stpyvista import stpyvista  # noqa: F401

        return True
    except ImportError:
        return False


def _mesh_to_polydata(
    mesh: Mesh,
    *,
    max_faces: int | None,
    seed: int,
):
    import pyvista as pv

    if mesh.faces.shape[0] == 0:
        v = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
        return pv.PolyData(v)

    nf = int(mesh.faces.shape[0])
    if max_faces is None or max_faces <= 0 or nf <= max_faces:
        display = mesh
    else:
        display = _subsample_mesh_faces(mesh, int(max_faces), seed)
    v = np.ascontiguousarray(display.vertices, dtype=np.float64)
    f = np.ascontiguousarray(display.faces, dtype=np.int64)
    faces_vtk = np.column_stack((np.full(len(f), 3, dtype=np.int64), f)).ravel()
    return pv.PolyData(v, faces_vtk)


def _polydata_set_point_rgb(surf, vertex_rgb: np.ndarray) -> None:
    """VTK point colors: RGB per vertex (interpolated across triangles with smooth shading)."""
    rgb = np.asarray(vertex_rgb, dtype=np.uint8)
    if rgb.shape[0] != surf.n_points:
        return
    surf.point_data["point_rgb"] = rgb
    for key in ("colors", "RGBA"):
        if key in surf.point_data:
            del surf.point_data[key]
    if "face_rgb" in surf.cell_data:
        del surf.cell_data["face_rgb"]


def streamlit_show_colored_bone(
    mesh: Mesh,
    vertex_rgb: np.ndarray,
    *,
    streamlit_key: str,
    mesh_path: Path | None = None,
    laplacian_iters: int = 0,
    max_faces: int | None = None,
) -> bool:
    """VTK surface with per-vertex RGB (shaded gradient across each triangle)."""
    try:
        import pyvista as pv
        from stpyvista import stpyvista
    except ImportError:
        return False

    if mesh.faces.shape[0] == 0:
        return False

    try:
        seed = _seed_from_path(mesh_path) if mesh_path is not None else 0
        surf = _mesh_to_polydata(mesh, max_faces=max_faces, seed=seed)
        if laplacian_iters > 0:
            surf = surf.smooth(
                n_iter=int(laplacian_iters),
                relaxation_factor=0.08,
                feature_smoothing=False,
                boundary_smoothing=False,
            )
        rgb = np.asarray(vertex_rgb, dtype=np.uint8)
        if rgb.shape[0] != surf.n_points:
            return False
        _polydata_set_point_rgb(surf, rgb)

        plotter = pv.Plotter(window_size=(920, 560))
        plotter.set_background("white")
        plotter.add_mesh(
            surf,
            scalars="point_rgb",
            rgb=True,
            smooth_shading=True,
            interpolate_before_map=True,
            show_edges=False,
        )
        plotter.reset_camera()
        stpyvista(plotter, key=streamlit_key)
    except Exception:
        return False
    return True


def streamlit_show_bone_and_plate(
    bone: Mesh,
    bone_vertex_rgb: np.ndarray,
    plate: Mesh,
    *,
    plate_color_hex: str,
    streamlit_key: str,
    bone_path: Path | None = None,
    plate_path: Path | None = None,
    laplacian_iters: int = 0,
    max_bone_faces: int | None = None,
    max_plate_faces: int | None = None,
) -> bool:
    try:
        import pyvista as pv
        from stpyvista import stpyvista
    except ImportError:
        return False

    if bone.faces.shape[0] == 0:
        return False

    try:
        seed_b = _seed_from_path(bone_path) if bone_path is not None else 0
        seed_p = _seed_from_path(plate_path) if plate_path is not None else seed_b + 7
        surf_b = _mesh_to_polydata(bone, max_faces=max_bone_faces, seed=seed_b)
        if laplacian_iters > 0:
            surf_b = surf_b.smooth(
                n_iter=int(laplacian_iters),
                relaxation_factor=0.08,
                feature_smoothing=False,
                boundary_smoothing=False,
            )
        rgb = np.asarray(bone_vertex_rgb, dtype=np.uint8)
        if rgb.shape[0] != surf_b.n_points:
            return False
        _polydata_set_point_rgb(surf_b, rgb)

        surf_p = _mesh_to_polydata(plate, max_faces=max_plate_faces, seed=seed_p)

        plotter = pv.Plotter(window_size=(920, 560))
        plotter.set_background("white")
        plotter.add_mesh(
            surf_b,
            scalars="point_rgb",
            rgb=True,
            smooth_shading=True,
            interpolate_before_map=True,
            show_edges=False,
            opacity=1.0,
        )
        plotter.add_mesh(
            surf_p,
            color=plate_color_hex,
            smooth_shading=True,
            show_edges=False,
            opacity=0.92,
        )
        plotter.reset_camera()
        stpyvista(plotter, key=streamlit_key)
    except Exception:
        return False
    return True


__all__ = [
    "PYVISTA_MAX_BONE_FACES",
    "PYVISTA_MAX_PLATE_FACES",
    "pyvista_stack_available",
    "streamlit_show_bone_and_plate",
    "streamlit_show_colored_bone",
]
