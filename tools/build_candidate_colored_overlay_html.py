"""Build one interactive HTML: label-colored reconstruction + plate + screw cylinders.

Screw cylinders use **per-vertex** color with ``flatshading=False`` (same smooth shading
as the TXT-colored bone in ``mesh_figure_colored_by_txt``), and a higher side count so
the shaft is rounder. Bone TXT coloring uses ``mode="smooth"`` to match the app.

Reads a miniplate candidate ``*.txt`` (JSON body) such as::

    .../_out_inferior_clean_batch_new/<CASE>/candidate_0001.txt

Uses reconstruction mesh + TXT from the Streamlit bundle (same case id as the
candidate folder) so paths stay portable. Plate mesh is loaded from the
candidate folder (``plate_mesh_straight_obj``). Does not modify ``visualizations.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from interference_app.visualizations import (  # noqa: E402
    Mesh,
    _maybe_cap_mesh_faces,
    _mesh3d_vertexcolor_trace,
    _seed_from_path,
    load_obj_mesh,
    load_point_cloud_json,
    save_figure_html,
    transfer_txt_colors_to_mesh_vertices,
)

_DEFAULT_SCREW_COLOR = "#3a3a3a"


def _hex_to_rgb_triplet(color: str) -> tuple[int, int, int]:
    c = color.strip()
    if c.startswith("#"):
        c = c[1:]
    if len(c) == 6 and all(ch in "0123456789abcdefABCDEF" for ch in c):
        return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    raise ValueError(f"--screw-color must be #RRGGBB, got {color!r}")


def _uniform_color_mesh3d_trace(
    mesh: Mesh,
    *,
    name: str,
    color_hex: str,
    opacity: float,
) -> go.Mesh3d:
    """Uniform color via per-vertex RGB; ``flatshading=False`` matches app bone traces."""
    r, g, b = _hex_to_rgb_triplet(color_hex)
    n = int(mesh.vertices.shape[0])
    vc = [f"rgb({r},{g},{b})"] * n
    return go.Mesh3d(
        x=mesh.vertices[:, 0],
        y=mesh.vertices[:, 1],
        z=mesh.vertices[:, 2],
        i=mesh.faces[:, 0],
        j=mesh.faces[:, 1],
        k=mesh.faces[:, 2],
        vertexcolor=vc,
        opacity=opacity,
        flatshading=False,
        name=name,
        showscale=False,
        hoverinfo="name",
    )


def _screw_cylinder_mesh(
    origin_mm: np.ndarray,
    tip_mm: np.ndarray,
    *,
    radius_mm: float,
    n_ring: int = 48,
) -> Mesh:
    """Closed cylinder along origin→tip (more ``n_ring`` → smoother shaft)."""
    o = np.asarray(origin_mm, dtype=np.float64).reshape(3)
    t = np.asarray(tip_mm, dtype=np.float64).reshape(3)
    axis = t - o
    ln = float(np.linalg.norm(axis))
    if ln < 1e-9:
        ln = 1e-9
    axis = axis / ln
    aux = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(axis, aux))) > 0.9:
        aux = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = np.cross(axis, aux)
    u = u / float(np.linalg.norm(u))
    v = np.cross(axis, u)
    angles = np.linspace(0.0, 2.0 * np.pi, num=int(n_ring), endpoint=False, dtype=np.float64)
    c = np.cos(angles)
    s = np.sin(angles)
    ring = radius_mm * (np.outer(c, u) + np.outer(s, v))
    bottom = o + ring
    top = t + ring
    vtx = np.vstack([bottom, top, o.reshape(1, 3), t.reshape(1, 3)])
    bi = 2 * n_ring
    c_bot = bi
    c_top = bi + 1
    faces: list[tuple[int, int, int]] = []
    for j in range(n_ring):
        jn = (j + 1) % n_ring
        faces.append((j, jn, n_ring + j))
        faces.append((jn, n_ring + jn, n_ring + j))
        faces.append((c_bot, jn, j))
        faces.append((c_top, n_ring + j, n_ring + jn))
    return Mesh(vertices=vtx, faces=np.asarray(faces, dtype=np.int64))


def _fig_global_limits(fig: go.Figure) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for tr in fig.data:
        if hasattr(tr, "x") and tr.x is not None:
            xs.extend([float(v) for v in tr.x if v is not None and not (isinstance(v, float) and np.isnan(v))])
        if hasattr(tr, "y") and tr.y is not None:
            ys.extend([float(v) for v in tr.y if v is not None and not (isinstance(v, float) and np.isnan(v))])
        if hasattr(tr, "z") and tr.z is not None:
            zs.extend([float(v) for v in tr.z if v is not None and not (isinstance(v, float) and np.isnan(v))])
    if not xs:
        return ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
    pad = 0.06
    def span(a: list[float]) -> tuple[float, float]:
        lo, hi = min(a), max(a)
        if lo == hi:
            lo, hi = lo - 1.0, hi + 1.0
        d = hi - lo
        return lo - pad * d, hi + pad * d
    return span(xs), span(ys), span(zs)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--candidate-txt",
        type=Path,
        required=True,
        help="Path to candidate_NNNN.txt (JSON) from miniplate pipeline output.",
    )
    ap.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    ap.add_argument(
        "--out-html",
        type=Path,
        default=None,
        help="Output HTML path (default: repo presentation_images/html/).",
    )
    ap.add_argument("--max-bone-faces", type=int, default=90_000)
    ap.add_argument("--bone-opacity", type=float, default=0.88)
    ap.add_argument("--plate-opacity", type=float, default=1.0)
    ap.add_argument(
        "--screw-radius-mm",
        type=float,
        default=0.0,
        help="Cylinder radius in mm; 0 = auto from screw length.",
    )
    ap.add_argument(
        "--screw-color",
        type=str,
        default=_DEFAULT_SCREW_COLOR,
        help="Hex color #RRGGBB for screw cylinders (default dark grey).",
    )
    ap.add_argument(
        "--screw-radius-scale",
        type=float,
        default=1.45,
        help="Multiply auto screw radius so grey cylinders extend past plate OBJ screw bosses.",
    )
    ap.add_argument(
        "--screw-rings",
        type=int,
        default=48,
        help="Vertices around each screw cylinder (higher → smoother outline).",
    )
    args = ap.parse_args(argv)

    cand_path = args.candidate_txt.resolve()
    repo = args.repo_root.resolve()
    case_id = cand_path.parent.name

    data = json.loads(cand_path.read_text(encoding="utf-8"))
    cand_id = int(data.get("candidate_id", 0))

    recon_obj = repo / "interference_app" / "output_data" / "reconstruction" / case_id / f"{case_id}.obj"
    recon_txt = repo / "interference_app" / "output_data" / "reconstruction" / case_id / f"{case_id}.txt"
    if not recon_txt.is_file():
        recon_txt = Path(str(data.get("bone_txt", "")))

    plate_rel = str(data.get("plate_mesh_straight_obj") or "")
    plate_obj = cand_path.parent / plate_rel if plate_rel else None

    if not recon_obj.is_file():
        raise SystemExit(f"Reconstruction OBJ not found: {recon_obj}")
    if not recon_txt.is_file():
        raise SystemExit(f"Reconstruction TXT not found: {recon_txt}")
    if plate_obj is None or not plate_obj.is_file():
        raise SystemExit(f"Straight plate OBJ not found: {plate_obj}")

    try:
        screw_hex = str(args.screw_color).strip()
        _hex_to_rgb_triplet(screw_hex)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    bone_mesh = load_obj_mesh(recon_obj)
    pc = load_point_cloud_json(recon_txt, max_points=None)
    v_rgb = transfer_txt_colors_to_mesh_vertices(
        bone_mesh,
        pc,
        mode="smooth",
        k_smooth=12,
        grey_vertices_not_in_txt_ids=True,
    )
    plate_mesh = load_obj_mesh(plate_obj)

    fig = go.Figure()
    seed = _seed_from_path(recon_obj)
    display_bone = _maybe_cap_mesh_faces(bone_mesh, int(args.max_bone_faces), seed)
    fig.add_trace(
        _mesh3d_vertexcolor_trace(
            display_bone,
            v_rgb,
            name="reconstruction (TXT-colored)",
            opacity=float(args.bone_opacity),
        )
    )

    fig.add_trace(
        _uniform_color_mesh3d_trace(
            plate_mesh,
            name=f"miniplate candidate_{cand_id:04d} (straight)",
            color_hex="#1565c0",
            opacity=float(args.plate_opacity),
        )
    )

    screws = data.get("screws_straight") or data.get("screws_world") or []
    for sc in screws:
        if not isinstance(sc, dict):
            continue
        hi = int(sc.get("hole_index", 0))
        o = np.asarray(sc.get("origin_mm", []), dtype=np.float64).ravel()[:3]
        t = np.asarray(sc.get("tip_mm", []), dtype=np.float64).ravel()[:3]
        if o.size < 3 or t.size < 3:
            continue
        seg_len = float(np.linalg.norm(t - o))
        rad = float(args.screw_radius_mm)
        if rad <= 0.0:
            rad = float(np.clip(0.12 * seg_len, 0.45, 1.8))
        rad *= float(args.screw_radius_scale)
        n_ring = int(np.clip(int(args.screw_rings), 8, 256))
        cyl = _screw_cylinder_mesh(o, t, radius_mm=rad, n_ring=n_ring)
        fig.add_trace(
            _uniform_color_mesh3d_trace(
                cyl,
                name=f"screw hole {hi}",
                color_hex=screw_hex,
                opacity=1.0,
            )
        )

    title = f"{case_id} — candidate_{cand_id:04d} (colored reconstruction + plate + screws)"
    fig.update_layout(
        title=title,
        scene=dict(
            aspectmode="data",
            xaxis=dict(title="X", visible=False),
            yaxis=dict(title="Y", visible=False),
            zaxis=dict(title="Z", visible=False),
            bgcolor="#f8f8f8",
        ),
        margin=dict(l=0, r=0, t=48, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        paper_bgcolor="#ffffff",
    )

    lims = _fig_global_limits(fig)
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=list(lims[0]), visible=False),
            yaxis=dict(range=list(lims[1]), visible=False),
            zaxis=dict(range=list(lims[2]), visible=False),
            aspectmode="cube",
        )
    )

    out = args.out_html
    if out is None:
        out = (
            repo
            / "presentation_images"
            / "html"
            / f"{case_id}_candidate_{cand_id:04d}_reconstruction_plate_screws.html"
        )
    out = Path(out).resolve()
    save_figure_html(fig, out)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
