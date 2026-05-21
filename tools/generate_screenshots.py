"""Generate nice-looking screenshots (PNGs) for docs/pitch.

This script uses the same plotting helpers as the Streamlit viewer to render:
- Segmentation point cloud (colored)
- Reconstruction mesh colored by reconstruction TXT labels
- Reconstruction + miniplate candidate overlay

Requires:
- ``kaleido`` (Plotly image export backend)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import plotly.graph_objects as go

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from interference_app.visualizations import (
    combined_figure,
    load_obj_mesh,
    load_point_cloud_json,
    mesh_figure_colored_by_txt,
    point_cloud_figure,
)


_CAMERAS: dict[str, dict] = {
    "iso": dict(eye=dict(x=1.35, y=1.35, z=1.15)),
    "front": dict(eye=dict(x=0.0, y=2.2, z=0.0)),
    "back": dict(eye=dict(x=0.0, y=-2.2, z=0.0)),
    "left": dict(eye=dict(x=-2.2, y=0.0, z=0.0)),
    "right": dict(eye=dict(x=2.2, y=0.0, z=0.0)),
    "top": dict(eye=dict(x=0.0, y=0.0, z=2.4)),
}


def _write_multiangle_pngs(
    fig: go.Figure,
    out_prefix: Path,
    *,
    width: int,
    height: int,
    scale: int,
    angles: tuple[str, ...],
) -> list[Path]:
    # Fail fast if kaleido is missing.
    try:
        import kaleido  # noqa: F401
    except Exception as exc:
        raise SystemExit(
            "PNG export requires kaleido. Install with: pip install kaleido"
        ) from exc

    written: list[Path] = []
    for a in angles:
        cam = _CAMERAS.get(a)
        if cam is None:
            continue
        fig_a = go.Figure(fig)
        fig_a.update_layout(scene_camera=cam)
        out_png = out_prefix.with_name(out_prefix.name + f"__{a}.png")
        fig_a.write_image(str(out_png), width=width, height=height, scale=scale)
        written.append(out_png)
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--case-id", type=str, default="BONE01_01")
    ap.add_argument("--candidate-id", type=int, default=0, help="Candidate ID to overlay (from summary.json).")
    ap.add_argument("--plate-label", type=str, default=None, help="Optional: subfolder under miniplate_positions/<case_id>/")
    ap.add_argument("--out-dir", type=Path, default=Path("screenshots_out"))
    ap.add_argument("--max-faces", type=int, default=90_000, help="Cap reconstruction faces for speed.")
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=1000)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument(
        "--angles",
        type=str,
        default="iso,front,left,right,top",
        help="Comma-separated camera angles: iso,front,back,left,right,top",
    )
    args = ap.parse_args(argv)

    repo_root: Path = args.repo_root.resolve()
    out_dir: Path = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    seg_txt = repo_root / "interference_app" / "output_data" / "segmentation" / args.case_id / f"{args.case_id}.txt"
    rec_obj = repo_root / "interference_app" / "output_data" / "reconstruction" / args.case_id / f"{args.case_id}.obj"
    rec_txt = repo_root / "interference_app" / "output_data" / "reconstruction" / args.case_id / f"{args.case_id}.txt"

    mp_root = repo_root / "interference_app" / "output_data" / "miniplate_positions" / args.case_id
    if args.plate_label:
        mp_root = mp_root / args.plate_label
    plate_obj = mp_root / "meshes" / f"candidate_{args.candidate_id:04d}_straight.obj"
    if not plate_obj.is_file():
        # Support filenames like:
        # - candidate_0001_original_straight.obj
        # - candidate_0001_mirrored_y_straight.obj
        # - candidate_0001.obj
        matches = sorted((mp_root / "meshes").glob(f"candidate_{args.candidate_id:04d}_*straight.obj"))
        if matches:
            plate_obj = matches[0]
        else:
            plate_obj = mp_root / "meshes" / f"candidate_{args.candidate_id:04d}.obj"

    angles = tuple(a.strip() for a in str(args.angles).split(",") if a.strip())
    written: list[Path] = []

    # 1) Segmentation
    if seg_txt.is_file():
        pc_seg = load_point_cloud_json(seg_txt, max_points=None)
        fig_seg = point_cloud_figure(pc_seg, title=f"{args.case_id} — segmentation (colored)")
        written.extend(
            _write_multiangle_pngs(
                fig_seg,
                out_dir / f"{args.case_id}_01_segmentation",
                width=args.width,
                height=args.height,
                scale=args.scale,
                angles=angles,
            )
        )

    # 2) Reconstruction (mesh + TXT colors)
    if rec_obj.is_file() and rec_txt.is_file():
        mesh = load_obj_mesh(rec_obj)
        pc_rec = load_point_cloud_json(rec_txt, max_points=None)
        fig_rec = mesh_figure_colored_by_txt(
            mesh,
            pc_rec,
            title=f"{args.case_id} — reconstruction (mesh, nearest TXT colors)",
            mesh_path=rec_obj,
            mode="nearest",
            k_smooth=12,
            grey_vertices_not_in_txt_ids=True,
            max_faces=args.max_faces,
        )
        written.extend(
            _write_multiangle_pngs(
                fig_rec,
                out_dir / f"{args.case_id}_02_reconstruction_mesh_txt",
                width=args.width,
                height=args.height,
                scale=args.scale,
                angles=angles,
            )
        )

    # 3) Overlay candidate plate
    if rec_obj.is_file() and plate_obj.is_file():
        bone_mesh = load_obj_mesh(rec_obj)
        plate_mesh = load_obj_mesh(plate_obj)
        fig_plate = combined_figure(
            bone_mesh=bone_mesh,
            plate_meshes={args.candidate_id: plate_mesh},
            selected_ids=[args.candidate_id],
            title=f"{args.case_id} — candidate_{args.candidate_id:04d} overlay",
            bone_path=rec_obj,
            highlight_ids=[args.candidate_id],
        )
        written.extend(
            _write_multiangle_pngs(
                fig_plate,
                out_dir / f"{args.case_id}_03_plate_overlay_c{args.candidate_id:04d}",
                width=args.width,
                height=args.height,
                scale=args.scale,
                angles=angles,
            )
        )

    for p in written:
        print(str(p))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

