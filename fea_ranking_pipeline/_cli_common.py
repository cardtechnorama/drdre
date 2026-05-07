"""Shared argparse helpers for pipeline mechanical knobs and bone defaults."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .fea_runner import DEFAULT_BONE_OBJ, DEFAULT_BONE_TXT, DEFAULT_MECHANICAL_FEM_DIR


def add_bone_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bone-txt", type=Path, default=DEFAULT_BONE_TXT)
    parser.add_argument("--bone-obj", type=Path, default=DEFAULT_BONE_OBJ)
    parser.add_argument(
        "--mechanical-fem-dir",
        type=Path,
        default=DEFAULT_MECHANICAL_FEM_DIR,
        help="Folder containing run_fea_hemijaw.py",
    )


def add_cache_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fea-mesh-cache", type=Path, default=None)
    parser.add_argument(
        "--rebuild-cache",
        dest="rebuild_cache",
        action="store_true",
        help="Force rebuilding the shared FEA mesh cache from the first candidate",
    )
    parser.add_argument("--stop-on-error", action="store_true", help="Stop on first failure")


def add_pipeline_args(parser: argparse.ArgumentParser) -> None:
    """Add the run_fea_hemijaw.run_pipeline mechanical/solver knobs."""
    g = parser.add_argument_group("pipeline parameters")
    g.add_argument("--subsample", type=int, default=None)
    g.add_argument("--poisson-depth", type=int, default=None)
    g.add_argument("--tet-pitch", type=float, default=None)
    g.add_argument("--max-tet-nodes", type=int, default=None)
    g.add_argument("--E-bone-GPa", type=float, default=None)
    g.add_argument("--nu-bone", type=float, default=None)
    g.add_argument("--bite-N", type=float, default=None)
    g.add_argument("--plate-stiffness-factor", type=float, default=None)
    g.add_argument("--fracture-E-factor", type=float, default=None)
    g.add_argument("--max-load-nodes", type=int, default=None)
    g.add_argument("--interface-half-width-mm", type=float, default=None)
    g.add_argument("--interface-trace-radius-mm", type=float, default=None)
    g.add_argument("--node-interface-band-mm", type=float, default=None)
    g.add_argument("--continuous-mesh", action="store_true", default=False)
    g.add_argument("--fracture-node-radius-mm", type=float, default=None)
    g.add_argument("--plate-node-radius-mm", type=float, default=None)
    g.add_argument("--screw-radius-mm", type=float, default=None)
    g.add_argument("--screw-shell-mm", type=float, default=None)
    g.add_argument("--screw-stiffness-factor", type=float, default=None)


_PIPELINE_ARG_MAP: tuple[tuple[str, str], ...] = (
    ("subsample", "subsample_cloud"),
    ("poisson_depth", "poisson_depth"),
    ("tet_pitch", "tet_pitch"),
    ("max_tet_nodes", "max_tet_nodes"),
    ("E_bone_GPa", "E_bone_GPa"),
    ("nu_bone", "nu_bone"),
    ("bite_N", "bite_force_N"),
    ("plate_stiffness_factor", "plate_stiffness_factor"),
    ("fracture_E_factor", "fracture_E_factor"),
    ("max_load_nodes", "max_load_nodes"),
    ("interface_half_width_mm", "interface_half_width_mm"),
    ("interface_trace_radius_mm", "interface_trace_radius_mm"),
    ("node_interface_band_mm", "node_interface_band_mm"),
    ("fracture_node_radius_mm", "fracture_node_radius_mm"),
    ("plate_node_radius_mm", "plate_node_radius_mm"),
    ("screw_radius_mm", "screw_radius_mm"),
    ("screw_shell_mm", "screw_shell_mm"),
    ("screw_stiffness_factor", "screw_stiffness_factor"),
)


def pipeline_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Return only user-provided kwargs so ``run_pipeline`` defaults are preserved."""
    kw: dict[str, Any] = {}
    for ns_name, py_name in _PIPELINE_ARG_MAP:
        val = getattr(args, ns_name, None)
        if val is not None:
            kw[py_name] = val
    if getattr(args, "continuous_mesh", False):
        kw["continuous_mesh"] = True
    return kw
