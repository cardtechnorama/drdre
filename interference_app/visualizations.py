"""Interactive 3D visualizations for pipeline stage artifacts.

Produces Plotly figures from segmented point clouds (JSON-TXT), recon-
structed meshes (OBJ), and miniplate candidate meshes (OBJ). Figures are
rendered inline inside Streamlit and can also be exported as standalone
HTML files for offline inspection / download.

Large point clouds and meshes are subsampled deterministically so the
client browser stays responsive. Sampling is seeded by a stable hash of
the input path so identical runs produce identical visuals.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import plotly.graph_objects as go
from scipy.spatial import cKDTree

MAX_POINTS_DEFAULT: int = 40_000
MAX_MESH_POINTS_DEFAULT: int = 35_000
MAX_MESH_FACES_DEFAULT: int = 60_000

COLOR_MAP: dict[str, str] = {
    "Red": "#F08080",
    "Orange": "#ffcc80",
    "Yellow": "#ffe082",
    "Green": "#66CDAA",
    "Purple": "#BA91FF",
    "Magenta": "#f48fb1",
    "Gray": "#AAAAAA",
    "Grey": "#AAAAAA",
    "White": "#fafafa",
    "Black": "#90a4ae",
    "Green+Yellow": "#dcedc8",
    "Red+Orange": "#ffccbc",
    "Purple+Red": "#ffcdd2",
}
UNKNOWN_COLOR: str = "#AAAAAA"
BONE_COLOR: str = "#e8e6df"

_PLATE_PALETTE: tuple[str, ...] = (
    "#90caf9",
    "#ffcc80",
    "#a5d6a7",
    "#ef9a9a",
    "#b39ddb",
    "#bcaaa4",
    "#f48fb1",
    "#bdbdbd",
    "#d4e157",
    "#80deea",
    "#9fa8da",
    "#c5e1a5",
)

PLATE_PALETTE: tuple[str, ...] = _PLATE_PALETTE


@dataclass(frozen=True)
class PointCloud:
    xyz: np.ndarray
    labels: np.ndarray
    #: If set, every ``Data[]`` row in the source TXT included a vertex ``ID``; mask is
    #: the set of indices that appear in the file. Mesh vertex indices not in this set
    #: (e.g. appended teeth) are treated as absent from the TXT when grey-mask is on.
    txt_vertex_ids: frozenset[int] | None = None


@dataclass(frozen=True)
class Mesh:
    vertices: np.ndarray
    faces: np.ndarray


def negate_x_points(xyz: np.ndarray) -> np.ndarray:
    """Flip X (TXT JSON frame vs OBJ export frame used across the pipeline)."""
    out = np.asarray(xyz, dtype=np.float32).copy()
    if out.size:
        out[..., 0] = -out[..., 0]
    return out


def mesh_negate_x(mesh: Mesh) -> Mesh:
    return Mesh(
        vertices=negate_x_points(mesh.vertices),
        faces=np.asarray(mesh.faces, dtype=np.int64).copy(),
    )


def point_cloud_negate_x(pc: PointCloud) -> PointCloud:
    return PointCloud(
        xyz=negate_x_points(pc.xyz),
        labels=pc.labels,
        txt_vertex_ids=pc.txt_vertex_ids,
    )


def _seed_from_path(path: Path) -> int:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def _subsample_indices(n: int, max_n: int, seed: int) -> np.ndarray:
    if n <= max_n:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_n, replace=False))


def _subsample_mesh_faces(mesh: Mesh, max_faces: int, seed: int) -> Mesh:
    """Keep every vertex; draw a random subset of triangles (still a mesh surface)."""
    fc = mesh.faces
    nf = int(fc.shape[0])
    if nf == 0 or nf <= max_faces:
        return mesh
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(nf, size=max_faces, replace=False))
    return Mesh(vertices=mesh.vertices, faces=fc[pick])


def _maybe_cap_mesh_faces(mesh: Mesh, max_faces: int | None, seed: int) -> Mesh:
    """``max_faces`` ``None`` or ``<= 0``: use every triangle (full OBJ surface)."""
    if max_faces is None or max_faces <= 0:
        return mesh
    return _subsample_mesh_faces(mesh, int(max_faces), seed)


def _coord_xyz(coord: Any) -> tuple[float, float, float]:
    if coord is None:
        return 0.0, 0.0, 0.0
    if isinstance(coord, dict):
        return (
            float(coord.get("x", coord.get("X", 0.0)) or 0.0),
            float(coord.get("y", coord.get("Y", 0.0)) or 0.0),
            float(coord.get("z", coord.get("Z", 0.0)) or 0.0),
        )
    try:
        return float(coord[0]), float(coord[1]), float(coord[2])
    except (IndexError, TypeError, ValueError):
        return 0.0, 0.0, 0.0


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.strip().lstrip("#")
    if len(h) >= 6 and all(c in "0123456789abcdefABCDEF" for c in h[:6]):
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (170, 170, 170)


def _labels_to_rgb_uint8(labels: np.ndarray) -> np.ndarray:
    """One RGB row per label string (0–255)."""
    lut: dict[str, int] = {}
    palette: list[tuple[int, int, int]] = []
    codes: list[int] = []
    for lab in labels.tolist():
        key = str(lab) if lab is not None else ""
        if key not in lut:
            h = COLOR_MAP.get(key, UNKNOWN_COLOR)
            lut[key] = len(palette)
            palette.append(_hex_to_rgb(h))
        codes.append(lut[key])
    pal = np.asarray(palette, dtype=np.uint8)
    return pal[np.asarray(codes, dtype=np.int64)]


def load_point_cloud_json(
    path: Path,
    max_points: int | None = MAX_POINTS_DEFAULT,
) -> PointCloud:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    data = raw.get("Data") if isinstance(raw, dict) else raw
    if not isinstance(data, list) or not data:
        return PointCloud(
            xyz=np.zeros((0, 3), dtype=np.float32),
            labels=np.array([], dtype=object),
            txt_vertex_ids=None,
        )

    txt_vertex_ids: frozenset[int] | None = None
    parsed_ids: list[int] = []
    for pt in data:
        raw_id = pt.get("ID") if isinstance(pt, dict) else None
        if raw_id is None:
            parsed_ids = []
            break
        try:
            parsed_ids.append(int(raw_id))
        except (TypeError, ValueError):
            parsed_ids = []
            break
    if parsed_ids:
        txt_vertex_ids = frozenset(parsed_ids)

    n = len(data)
    seed = _seed_from_path(Path(path))
    if max_points is None or n <= max_points:
        idx = np.arange(n, dtype=np.int64)
    else:
        idx = _subsample_indices(n, max_points, seed)
    xyz = np.empty((idx.size, 3), dtype=np.float32)
    labels = np.empty(idx.size, dtype=object)
    for out_i, src_i in enumerate(idx.tolist()):
        pt = data[src_i]
        x, y, z = _coord_xyz(pt.get("Coord"))
        xyz[out_i, 0] = x
        xyz[out_i, 1] = y
        xyz[out_i, 2] = z
        lab = pt.get("Color", "")
        labels[out_i] = str(lab) if lab else ""
    return PointCloud(xyz=xyz, labels=labels, txt_vertex_ids=txt_vertex_ids)


def _vertex_grey_mask_not_in_txt_ids(
    n_verts: int, txt_vertex_ids: frozenset[int] | None
) -> np.ndarray:
    """``True`` where the mesh vertex index does not appear in ``Data[].ID``."""
    if txt_vertex_ids is None or n_verts <= 0:
        return np.zeros(n_verts, dtype=bool)
    listed = np.fromiter(
        (i for i in txt_vertex_ids if 0 <= i < n_verts),
        dtype=np.int64,
    )
    grey = np.ones(n_verts, dtype=bool)
    if listed.size:
        grey[listed] = False
    return grey


def transfer_txt_colors_to_mesh_vertices(
    mesh: Mesh,
    pc: PointCloud,
    *,
    mode: str = "smooth",
    k_smooth: int = 10,
    eps: float = 1e-4,
    grey_vertices_not_in_txt_ids: bool = True,
) -> np.ndarray:
    """Map classified TXT points onto mesh vertices (bone mesh has more verts than TXT).

    ``mode``:
      - ``nearest``: same RGB as nearest TXT point (sharp class boundaries).
      - ``smooth``: inverse-distance weighted blend of ``k_smooth`` neighbors
        (smooth transitions where labels mix near teeth gaps / fracture margins).

    When ``grey_vertices_not_in_txt_ids`` is true and ``pc.txt_vertex_ids`` is set
    (every TXT row had an ``ID``), mesh vertex indices **not** listed in the TXT
    (e.g. appended teeth) are painted ``UNKNOWN_COLOR`` (grey). If the TXT omits
    ``ID`` on any row, ``txt_vertex_ids`` is unset and no ID-based greying applies.

    Returns ``(n_verts, 3)`` uint8 RGB rows.
    """
    nv = int(mesh.vertices.shape[0])
    if nv == 0 or pc.xyz.shape[0] == 0:
        g = np.asarray(_hex_to_rgb(UNKNOWN_COLOR), dtype=np.uint8).reshape(1, 3)
        return np.tile(g, (max(nv, 0), 1))
    if mode not in ("nearest", "smooth"):
        raise ValueError("mode must be 'nearest' or 'smooth'")

    tree = cKDTree(np.asarray(pc.xyz, dtype=np.float64))
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    rgb_pts = _labels_to_rgb_uint8(pc.labels)
    grey_rgb = np.asarray(_hex_to_rgb(UNKNOWN_COLOR), dtype=np.uint8)
    grey_v = (
        _vertex_grey_mask_not_in_txt_ids(nv, pc.txt_vertex_ids)
        if grey_vertices_not_in_txt_ids
        else np.zeros(nv, dtype=bool)
    )

    if mode == "nearest":
        _, idx = tree.query(verts, k=1, workers=-1)
        idx = np.asarray(idx, dtype=np.int64).reshape(-1)
        out = rgb_pts[idx].copy()
        out[grey_v] = grey_rgb
        return out

    k_req = min(max(2, int(k_smooth)), int(pc.xyz.shape[0]))
    dists, idx = tree.query(verts, k=k_req, workers=-1)
    dists = np.asarray(dists, dtype=np.float64)
    idx = np.asarray(idx, dtype=np.int64)
    if k_req == 1:
        out = rgb_pts[idx[:, 0]].copy()
    else:
        if dists.ndim == 1:
            dists = dists.reshape(-1, 1)
            idx = idx.reshape(-1, 1)
        w = 1.0 / (np.square(dists) + eps)
        w /= np.maximum(np.sum(w, axis=1, keepdims=True), 1e-12)
        gathered = rgb_pts[idx]
        blended = np.sum(gathered.astype(np.float64) * w[..., np.newaxis], axis=1)
        out = np.clip(np.round(blended), 0, 255).astype(np.uint8)
    out[grey_v] = grey_rgb
    return out


def _mesh3d_vertexcolor_trace(
    mesh: Mesh,
    vertex_rgb: np.ndarray,
    *,
    name: str,
    opacity: float,
) -> go.Mesh3d:
    """Per-vertex RGB; Plotly interpolates (Gouraud-style) across each triangle."""
    rgb = np.asarray(vertex_rgb, dtype=np.float64)
    vertexcolor = [f"rgb({int(r)},{int(g)},{int(b)})" for r, g, b in rgb]
    return go.Mesh3d(
        x=mesh.vertices[:, 0],
        y=mesh.vertices[:, 1],
        z=mesh.vertices[:, 2],
        i=mesh.faces[:, 0],
        j=mesh.faces[:, 1],
        k=mesh.faces[:, 2],
        vertexcolor=vertexcolor,
        opacity=opacity,
        flatshading=False,
        name=name,
        showscale=False,
        hoverinfo="name",
    )


def _mesh_as_points_colored_trace(
    verts: np.ndarray,
    vertex_rgb: np.ndarray,
    *,
    name: str,
    marker_size: float,
    seed: int,
    max_points: int,
    mesh_path: Path | None,
) -> go.Scatter3d:
    n = int(verts.shape[0])
    v = verts
    rh = vertex_rgb
    if n > max_points:
        idx = _subsample_indices(n, max_points, _seed_from_path(mesh_path) if mesh_path else seed)
        v = v[idx]
        rh = rh[idx]
    marker_colors = [f"rgb({int(r)},{int(g)},{int(b)})" for r, g, b in rh]
    return go.Scatter3d(
        x=v[:, 0],
        y=v[:, 1],
        z=v[:, 2],
        mode="markers",
        marker=dict(size=marker_size, color=marker_colors, opacity=0.78),
        name=f"{name} ({v.shape[0]} pts)",
        hovertemplate=f"{name}<br>x=%{{x:.2f}} y=%{{y:.2f}} z=%{{z:.2f}}<extra></extra>",
    )


def mesh_figure_colored_by_txt(
    mesh: Mesh,
    pc: PointCloud,
    *,
    title: str,
    mesh_path: Path | None = None,
    mode: str = "smooth",
    k_smooth: int = 10,
    grey_vertices_not_in_txt_ids: bool = True,
    max_faces: int | None = None,
    max_points: int = MAX_MESH_POINTS_DEFAULT,
) -> go.Figure:
    """Reconstructed bone as Mesh3d with per-vertex TXT colors (interpolated on each Δ).

    By default draws **all** OBJ triangles so the full surface is covered. Set ``max_faces``
    to a positive limit only if the browser struggles (that subsamples triangles).
    """
    v_rgb = transfer_txt_colors_to_mesh_vertices(
        mesh,
        pc,
        mode=mode,
        k_smooth=k_smooth,
        grey_vertices_not_in_txt_ids=grey_vertices_not_in_txt_ids,
    )
    fig = go.Figure()
    seed = _seed_from_path(mesh_path) if mesh_path is not None else 0
    nf_total = int(mesh.faces.shape[0])
    if nf_total == 0:
        fig.add_trace(
            _mesh_as_points_colored_trace(
                np.asarray(mesh.vertices, dtype=np.float32),
                v_rgb,
                name="bone (vertices only — OBJ has no faces)",
                marker_size=1.25,
                seed=seed,
                max_points=max_points,
                mesh_path=mesh_path,
            )
        )
    else:
        display_mesh = _maybe_cap_mesh_faces(mesh, max_faces, seed)
        nf = int(display_mesh.faces.shape[0])
        trace_name = "bone (TXT-colored mesh, full surface)"
        if nf < nf_total:
            trace_name = f"bone (TXT mesh · {nf:,} / {nf_total:,} Δ — subsampled)"
        fig.add_trace(
            _mesh3d_vertexcolor_trace(
                display_mesh,
                v_rgb,
                name=trace_name,
                opacity=1.0,
            )
        )
    _apply_scene_layout(fig, title)
    return fig


def _parse_face_tokens(tokens: Sequence[str]) -> list[int]:
    idxs: list[int] = []
    for tok in tokens:
        head = tok.split("/", 1)[0]
        if not head:
            continue
        idxs.append(int(head) - 1)
    return idxs


def load_obj_mesh(path: Path) -> Mesh:
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line or line[0] == "#":
                continue
            parts = line.split()
            if not parts:
                continue
            tag = parts[0]
            if tag == "v" and len(parts) >= 4:
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif tag == "f" and len(parts) >= 4:
                idxs = _parse_face_tokens(parts[1:])
                if len(idxs) < 3:
                    continue
                for k in range(1, len(idxs) - 1):
                    faces.append((idxs[0], idxs[k], idxs[k + 1]))
    v = np.asarray(verts, dtype=np.float32).reshape(-1, 3)
    fc = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if fc.size:
        valid = (fc >= 0).all(axis=1) & (fc < v.shape[0]).all(axis=1)
        fc = fc[valid]
    return Mesh(vertices=v, faces=fc)


def _apply_scene_layout(fig: go.Figure, title: str) -> None:
    fig.update_layout(
        title=title,
        scene=dict(
            aspectmode="data",
            xaxis=dict(title="X"),
            yaxis=dict(title="Y"),
            zaxis=dict(title="Z"),
            bgcolor="#f8f8f8",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="top", y=1.05, xanchor="left", x=0.0),
        paper_bgcolor="#ffffff",
    )


def point_cloud_figure(
    pc: PointCloud,
    *,
    title: str,
    marker_size: float = 1.6,
) -> go.Figure:
    fig = go.Figure()
    if pc.xyz.size == 0:
        _apply_scene_layout(fig, title)
        return fig
    unique_labels = sorted({str(l) for l in pc.labels.tolist()})
    for lab in unique_labels:
        mask = pc.labels == lab
        if not mask.any():
            continue
        color = COLOR_MAP.get(lab, UNKNOWN_COLOR)
        display_name = lab if lab else "Unlabeled"
        fig.add_trace(
            go.Scatter3d(
                x=pc.xyz[mask, 0],
                y=pc.xyz[mask, 1],
                z=pc.xyz[mask, 2],
                mode="markers",
                marker=dict(size=marker_size, color=color, opacity=0.78),
                name=f"{display_name} ({int(mask.sum())})",
                hovertemplate=f"{display_name}<br>x=%{{x:.2f}} y=%{{y:.2f}} z=%{{z:.2f}}<extra></extra>",
            )
        )
    _apply_scene_layout(fig, title)
    return fig


def _mesh_as_points_trace(
    verts: np.ndarray,
    *,
    name: str,
    color: str,
    marker_size: float,
    seed: int,
    max_points: int,
) -> go.Scatter3d:
    if verts.shape[0] > max_points:
        idx = _subsample_indices(verts.shape[0], max_points, seed)
        verts = verts[idx]
    return go.Scatter3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        mode="markers",
        marker=dict(size=marker_size, color=color, opacity=0.72),
        name=f"{name} ({verts.shape[0]} pts)",
        hovertemplate=f"{name}<br>x=%{{x:.2f}} y=%{{y:.2f}} z=%{{z:.2f}}<extra></extra>",
    )


def _mesh3d_trace(
    mesh: Mesh,
    *,
    name: str,
    color: str,
    opacity: float,
) -> go.Mesh3d:
    return go.Mesh3d(
        x=mesh.vertices[:, 0],
        y=mesh.vertices[:, 1],
        z=mesh.vertices[:, 2],
        i=mesh.faces[:, 0],
        j=mesh.faces[:, 1],
        k=mesh.faces[:, 2],
        color=color,
        opacity=opacity,
        flatshading=True,
        name=name,
        showscale=False,
        hoverinfo="name",
    )


def mesh_figure(
    mesh: Mesh,
    *,
    title: str,
    mesh_path: Path | None = None,
    color: str = BONE_COLOR,
    max_faces: int = MAX_MESH_FACES_DEFAULT,
    max_points: int = MAX_MESH_POINTS_DEFAULT,
) -> go.Figure:
    """Return a point-cloud scatter for large meshes, a Mesh3d for small ones.

    Rendering a multi-hundred-thousand-triangle mesh via WebGL Mesh3d in
    the browser can lock up the page; we fall back to vertex-only
    scatter3d when the face count exceeds ``max_faces``.
    """
    fig = go.Figure()
    seed = _seed_from_path(mesh_path) if mesh_path is not None else 0
    if mesh.faces.shape[0] == 0 or mesh.faces.shape[0] > max_faces:
        fig.add_trace(
            _mesh_as_points_trace(
                mesh.vertices,
                name="mesh",
                color=color,
                marker_size=1.4,
                seed=seed,
                max_points=max_points,
            )
        )
    else:
        fig.add_trace(
            _mesh3d_trace(mesh, name="mesh", color=color, opacity=1.0)
        )
    _apply_scene_layout(fig, title)
    return fig


def candidates_overview_figure(
    bone_mesh: Mesh,
    plate_meshes: dict[int, Mesh],
    *,
    title: str,
    bone_path: Path | None = None,
    max_bone_points: int = MAX_MESH_POINTS_DEFAULT,
) -> go.Figure:
    """Render the reconstructed bone plus every candidate plate in one view."""
    return combined_figure(
        bone_mesh=bone_mesh,
        plate_meshes=plate_meshes,
        selected_ids=list(plate_meshes.keys()),
        title=title,
        bone_path=bone_path,
        max_bone_points=max_bone_points,
    )


def combined_figure(
    bone_mesh: Mesh,
    plate_meshes: dict[int, Mesh],
    selected_ids: Iterable[int],
    *,
    title: str,
    bone_path: Path | None = None,
    max_bone_points: int = MAX_MESH_POINTS_DEFAULT,
    bone_opacity: float = 0.35,
    plate_opacity: float = 0.95,
    highlight_ids: Iterable[int] | None = None,
) -> go.Figure:
    """Compose bone (semi-transparent) with the selected plate candidates.

    ``highlight_ids`` optionally emphasizes a subset of the selected
    plates (e.g., the top-N ranked) by increasing their opacity and
    placing them first in the legend.
    """
    fig = go.Figure()
    seed = _seed_from_path(bone_path) if bone_path is not None else 0
    fig.add_trace(
        _mesh_as_points_trace(
            bone_mesh.vertices,
            name="bone",
            color=BONE_COLOR,
            marker_size=1.1,
            seed=seed,
            max_points=max_bone_points,
        )
    )
    sel = list(dict.fromkeys(int(c) for c in selected_ids))
    hl = set(int(h) for h in (highlight_ids or []))
    sel.sort(key=lambda c: (c not in hl, c))
    for i, cid in enumerate(sel):
        m = plate_meshes.get(cid)
        if m is None or m.vertices.size == 0:
            continue
        color = _PLATE_PALETTE[i % len(_PLATE_PALETTE)]
        name = f"candidate_{cid:04d}"
        if cid in hl:
            name = f"[top] {name}"
        if m.faces.shape[0] > 0:
            fig.add_trace(
                _mesh3d_trace(m, name=name, color=color, opacity=plate_opacity)
            )
        else:
            fig.add_trace(
                _mesh_as_points_trace(
                    m.vertices,
                    name=name,
                    color=color,
                    marker_size=2.4,
                    seed=_seed_from_path(Path(name)),
                    max_points=max_bone_points,
                )
            )
    fig.data[0].opacity = bone_opacity
    _apply_scene_layout(fig, title)
    return fig


def combined_figure_colored_bone_txt(
    bone_pc: PointCloud,
    plate_meshes: dict[int, Mesh],
    selected_ids: Iterable[int],
    *,
    title: str,
    bone_path: Path | None = None,
    bone_marker_size: float = 1.15,
    bone_opacity: float = 0.42,
    plate_opacity: float = 0.95,
    highlight_ids: Iterable[int] | None = None,
    max_bone_points: int | None = MAX_POINTS_DEFAULT,
    plate_names: dict[int, str] | None = None,
    plate_colors: dict[int, str] | None = None,
) -> go.Figure:
    """Bone from classified TXT (per-point ``Color`` labels); optional mesh overlays."""
    fig = go.Figure()
    seed = _seed_from_path(bone_path) if bone_path is not None else 0
    xyz = bone_pc.xyz
    labels = bone_pc.labels
    if xyz.shape[0] == 0:
        _apply_scene_layout(fig, title)
        return fig
    if max_bone_points is not None and xyz.shape[0] > max_bone_points:
        idx = _subsample_indices(xyz.shape[0], max_bone_points, seed)
        xyz = xyz[idx]
        labels = labels[idx]
    unique_labels = sorted({str(l) for l in labels.tolist()})
    for lab in unique_labels:
        mask = labels == lab
        if not mask.any():
            continue
        color = COLOR_MAP.get(lab, UNKNOWN_COLOR)
        display_name = lab if lab else "Unlabeled"
        fig.add_trace(
            go.Scatter3d(
                x=xyz[mask, 0],
                y=xyz[mask, 1],
                z=xyz[mask, 2],
                mode="markers",
                marker=dict(
                    size=bone_marker_size,
                    color=color,
                    opacity=bone_opacity,
                ),
                name=f"bone {display_name}",
                hovertemplate=(
                    f"{display_name}<br>x=%{{x:.2f}} y=%{{y:.2f}} z=%{{z:.2f}}<extra></extra>"
                ),
            )
        )
    sel = list(dict.fromkeys(int(c) for c in selected_ids))
    hl = set(int(h) for h in (highlight_ids or []))
    sel.sort(key=lambda c: (c not in hl, c))
    max_plate_pts = MAX_MESH_POINTS_DEFAULT
    for i, cid in enumerate(sel):
        m = plate_meshes.get(cid)
        if m is None or m.vertices.size == 0:
            continue
        color = (plate_colors or {}).get(cid, _PLATE_PALETTE[i % len(_PLATE_PALETTE)])
        name = (plate_names or {}).get(cid, f"candidate_{cid:04d}")
        if cid in hl:
            name = f"[top] {name}"
        if m.faces.shape[0] > 0:
            fig.add_trace(
                _mesh3d_trace(m, name=name, color=color, opacity=plate_opacity)
            )
        else:
            fig.add_trace(
                _mesh_as_points_trace(
                    m.vertices,
                    name=name,
                    color=color,
                    marker_size=2.4,
                    seed=_seed_from_path(Path(name)),
                    max_points=max_plate_pts,
                )
            )
    _apply_scene_layout(fig, title)
    return fig


def combined_figure_colored_bone_mesh_txt(
    bone_mesh: Mesh,
    bone_vertex_rgb: np.ndarray,
    plate_meshes: dict[int, Mesh],
    selected_ids: Iterable[int],
    *,
    title: str,
    bone_path: Path | None = None,
    bone_opacity: float = 0.88,
    plate_opacity: float = 0.95,
    highlight_ids: Iterable[int] | None = None,
    max_bone_points: int = MAX_MESH_POINTS_DEFAULT,
    max_bone_faces: int | None = None,
) -> go.Figure:
    """Bone mesh with per-vertex TXT colors (interpolated on triangles) + plate meshes.

    ``max_bone_faces`` ``None`` (default): every bone triangle is drawn (full surface).
    """
    fig = go.Figure()
    seed = _seed_from_path(bone_path) if bone_path is not None else 0
    nf_total = int(bone_mesh.faces.shape[0])
    if nf_total == 0:
        fig.add_trace(
            _mesh_as_points_colored_trace(
                np.asarray(bone_mesh.vertices, dtype=np.float32),
                bone_vertex_rgb,
                name="bone (mesh verts — no faces in OBJ)",
                marker_size=1.1,
                seed=seed,
                max_points=max_bone_points,
                mesh_path=bone_path,
            )
        )
    else:
        display_bone = _maybe_cap_mesh_faces(bone_mesh, max_bone_faces, seed)
        nf = int(display_bone.faces.shape[0])
        trace_name = "bone (TXT-colored mesh, full surface)"
        if nf < nf_total:
            trace_name = f"bone (TXT mesh · {nf:,} / {nf_total:,} Δ — subsampled)"
        fig.add_trace(
            _mesh3d_vertexcolor_trace(
                display_bone,
                bone_vertex_rgb,
                name=trace_name,
                opacity=bone_opacity,
            )
        )
    sel = list(dict.fromkeys(int(c) for c in selected_ids))
    hl = set(int(h) for h in (highlight_ids or []))
    sel.sort(key=lambda c: (c not in hl, c))
    max_plate_pts = MAX_MESH_POINTS_DEFAULT
    for i, cid in enumerate(sel):
        m = plate_meshes.get(cid)
        if m is None or m.vertices.size == 0:
            continue
        color = _PLATE_PALETTE[i % len(_PLATE_PALETTE)]
        name = f"candidate_{cid:04d}"
        if cid in hl:
            name = f"[top] {name}"
        if m.faces.shape[0] > 0:
            fig.add_trace(
                _mesh3d_trace(m, name=name, color=color, opacity=plate_opacity)
            )
        else:
            fig.add_trace(
                _mesh_as_points_trace(
                    m.vertices,
                    name=name,
                    color=color,
                    marker_size=2.4,
                    seed=_seed_from_path(Path(name)),
                    max_points=max_plate_pts,
                )
            )
    _apply_scene_layout(fig, title)
    return fig


AIMER_A_COLOR = "#e74c3c"
AIMER_B_COLOR = "#1f77b4"


def combined_figure_bone_and_aimers(
    bone_mesh: Mesh,
    aimer_a: Mesh,
    aimer_b: Mesh,
    *,
    title: str,
    bone_vertex_rgb: np.ndarray | None = None,
    bone_path: Path | None = None,
    bone_opacity: float = 0.82,
    aimer_opacity: float = 0.92,
    max_bone_faces: int | None = None,
) -> go.Figure:
    """Input bone with surgical aimer shells A and B."""
    fig = go.Figure()
    seed = _seed_from_path(bone_path) if bone_path is not None else 0
    nf_total = int(bone_mesh.faces.shape[0])
    if nf_total == 0:
        verts = np.asarray(bone_mesh.vertices, dtype=np.float32)
        if bone_vertex_rgb is not None and bone_vertex_rgb.shape[0] == verts.shape[0]:
            fig.add_trace(
                _mesh_as_points_colored_trace(
                    verts,
                    bone_vertex_rgb,
                    name="bone",
                    marker_size=1.1,
                    seed=seed,
                    max_points=MAX_MESH_POINTS_DEFAULT,
                    mesh_path=bone_path,
                )
            )
        else:
            fig.add_trace(
                _mesh_as_points_trace(
                    verts,
                    name="bone",
                    color=BONE_COLOR,
                    marker_size=1.1,
                    seed=seed,
                    max_points=MAX_MESH_POINTS_DEFAULT,
                )
            )
    else:
        display_bone = _maybe_cap_mesh_faces(bone_mesh, max_bone_faces, seed)
        if bone_vertex_rgb is not None and bone_vertex_rgb.shape[0] == bone_mesh.vertices.shape[0]:
            fig.add_trace(
                _mesh3d_vertexcolor_trace(
                    display_bone,
                    bone_vertex_rgb,
                    name="bone (segmentation)",
                    opacity=bone_opacity,
                )
            )
        else:
            fig.add_trace(
                _mesh3d_trace(
                    display_bone,
                    name="bone (segmentation)",
                    color=BONE_COLOR,
                    opacity=bone_opacity,
                )
            )
    for label, mesh, color in (
        ("aimer A", aimer_a, AIMER_A_COLOR),
        ("aimer B", aimer_b, AIMER_B_COLOR),
    ):
        if mesh.vertices.size == 0:
            continue
        if mesh.faces.shape[0] > 0:
            fig.add_trace(
                _mesh3d_trace(mesh, name=label, color=color, opacity=aimer_opacity)
            )
        else:
            fig.add_trace(
                _mesh_as_points_trace(
                    mesh.vertices,
                    name=label,
                    color=color,
                    marker_size=2.6,
                    seed=seed + (1 if label.endswith("A") else 2),
                    max_points=MAX_MESH_POINTS_DEFAULT,
                )
            )
    _apply_scene_layout(fig, title)
    return fig


def save_figure_html(fig: go.Figure, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)
    return path


__all__ = [
    "BONE_COLOR",
    "COLOR_MAP",
    "MAX_MESH_FACES_DEFAULT",
    "MAX_MESH_POINTS_DEFAULT",
    "MAX_POINTS_DEFAULT",
    "Mesh",
    "mesh_negate_x",
    "negate_x_points",
    "point_cloud_negate_x",
    "PLATE_PALETTE",
    "PointCloud",
    "candidates_overview_figure",
    "AIMER_A_COLOR",
    "AIMER_B_COLOR",
    "combined_figure",
    "combined_figure_bone_and_aimers",
    "combined_figure_colored_bone_mesh_txt",
    "combined_figure_colored_bone_txt",
    "load_obj_mesh",
    "load_point_cloud_json",
    "mesh_figure_colored_by_txt",
    "transfer_txt_colors_to_mesh_vertices",
    "point_cloud_figure",
    "save_figure_html",
]
