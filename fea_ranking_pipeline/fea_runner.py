"""Hemijaw FEA batch/selected runner, decoupled from the mechanical pipeline CLI."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

DEFAULT_MECHANICAL_FEM_DIR = Path(__file__).resolve().parent.parent / "mechanical_fem_pipeline"

DEFAULT_BONE_TXT = Path(
    r"T:\card\dr.dre miniplate\Exported\reconstructed_mesh_vote\BONE_01_EDIT_01.txt"
)
DEFAULT_BONE_OBJ = Path(
    r"T:\card\dr.dre miniplate\Exported\reconstructed_mesh_vote\BONE_01_EDIT_01.obj"
)

_CANDIDATE_STEM_RE = re.compile(r"^(candidate[_-]?\d+)(?:_straight|_bent)?$", re.IGNORECASE)
_CANDIDATE_ID_RE = re.compile(r"candidate[_-]?(\d+)", re.IGNORECASE)


def candidate_name_from_plate(path: Path) -> str:
    """Normalize a plate OBJ filename into a canonical ``candidate_XXXX`` name."""
    stem = Path(path).stem
    m = _CANDIDATE_STEM_RE.match(stem)
    return m.group(1).lower() if m else stem


def candidate_id_from_name(name: str) -> int | None:
    m = _CANDIDATE_ID_RE.search(name)
    return int(m.group(1)) if m else None


def resolve_plate_obj(mesh_dir: Path, candidate_id: int) -> Path:
    """Locate the straight plate OBJ for a candidate id in ``mesh_dir``."""
    mesh_dir = Path(mesh_dir)
    stem = f"candidate_{int(candidate_id):04d}"
    exact = list(mesh_dir.glob(f"{stem}_straight.obj"))
    if len(exact) == 1:
        return exact[0]
    loose = [p for p in mesh_dir.glob(f"{stem}*straight.obj") if p.is_file()]
    if len(loose) == 1:
        return loose[0]
    if not exact and not loose:
        raise FileNotFoundError(f"No straight plate OBJ for candidate {candidate_id} in {mesh_dir}")
    found = exact or loose
    raise RuntimeError(f"Expected exactly one OBJ for candidate {candidate_id}, found {len(found)}: {found}")


def _import_run_pipeline(mechanical_fem_dir: Path) -> Callable[..., dict]:
    p = str(Path(mechanical_fem_dir).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)
    from run_fea_hemijaw import run_pipeline  # type: ignore

    return run_pipeline


@dataclass(frozen=True)
class FEARunConfig:
    bone_txt: Path
    bone_obj: Path
    mesh_dir: Path
    out_root: Path
    mechanical_fem_dir: Path = DEFAULT_MECHANICAL_FEM_DIR
    fea_mesh_cache: Path | None = None
    rebuild_cache: bool = False
    stop_on_error: bool = False
    pipeline_kwargs: dict[str, Any] = field(default_factory=dict)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _run_plates(
    plates: Sequence[Path],
    config: FEARunConfig,
    log: Callable[[str], None],
    *,
    extra_meta: dict[str, Any] | None = None,
) -> dict:
    """Shared loop: run FEA for each plate OBJ, reusing the shared mesh cache."""
    run_pipeline = _import_run_pipeline(config.mechanical_fem_dir)

    out_root = Path(config.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    cache = (
        Path(config.fea_mesh_cache).resolve()
        if config.fea_mesh_cache
        else (out_root / "bone_fea_mesh.npz")
    )
    need_build = bool(config.rebuild_cache or (not cache.is_file()))

    results: list[dict] = []
    failures: list[dict] = []
    total = len(plates)

    for idx, plate_obj in enumerate(plates, start=1):
        plate_obj = Path(plate_obj).resolve()
        candidate = candidate_name_from_plate(plate_obj)
        cid = candidate_id_from_name(candidate)
        out_dir = out_root / candidate
        fea_result_path = out_dir / "fea_result.json"

        log(f"[{idx}/{total}] {plate_obj.name}")
        try:
            result = run_pipeline(
                bone_txt=Path(config.bone_txt).resolve(),
                bone_obj=Path(config.bone_obj).resolve(),
                plate_obj=plate_obj,
                out_dir=out_dir,
                fea_mesh_path=None if need_build else cache,
                export_fea_mesh_path=cache if need_build else None,
                **config.pipeline_kwargs,
            )
            need_build = False
            entry: dict[str, Any] = {
                "candidate": candidate,
                "plate_obj": str(plate_obj),
                "out_dir": str(out_dir),
                "fea_result_json": str(fea_result_path),
                "result": result,
            }
            if cid is not None:
                entry["candidate_id"] = cid
            results.append(entry)
        except Exception as exc:  # noqa: BLE001
            err: dict[str, Any] = {
                "candidate": candidate,
                "plate_obj": str(plate_obj),
                "out_dir": str(out_dir),
                "error": f"{type(exc).__name__}: {exc}",
            }
            if cid is not None:
                err["candidate_id"] = cid
            failures.append(err)
            if config.stop_on_error:
                break

    payload: dict[str, Any] = {
        "created_at_utc": _now_iso(),
        "mesh_dir": str(Path(config.mesh_dir).resolve()),
        "bone_txt": str(Path(config.bone_txt).resolve()),
        "bone_obj": str(Path(config.bone_obj).resolve()),
        "out_root": str(out_root),
        "fea_mesh_cache": str(cache) if cache.is_file() else None,
        "n_total": total,
        "n_success": len(results),
        "n_failed": len(failures),
        "results": results,
        "failures": failures,
    }
    if extra_meta:
        payload.update(extra_meta)
    return payload


def run_selected(
    candidate_ids: Iterable[int],
    config: FEARunConfig,
    log: Callable[[str], None] = print,
) -> dict:
    """Run FEA for a user-specified list of candidate IDs."""
    ids = [int(x) for x in candidate_ids]
    plates: list[Path] = []
    missing: list[dict] = []
    for cid in ids:
        try:
            plates.append(resolve_plate_obj(config.mesh_dir, cid))
        except Exception as exc:  # noqa: BLE001
            missing.append(
                {
                    "candidate_id": cid,
                    "candidate": f"candidate_{cid:04d}",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    payload = _run_plates(
        plates,
        config,
        log,
        extra_meta={"selected_candidate_ids": ids},
    )
    if missing:
        payload["failures"] = list(missing) + list(payload["failures"])
        payload["n_failed"] = len(payload["failures"])
        payload["n_total"] = len(ids)
    return payload


def run_glob(
    plate_glob: str,
    config: FEARunConfig,
    log: Callable[[str], None] = print,
) -> dict:
    """Run FEA for every plate OBJ matching ``plate_glob`` under ``config.mesh_dir``."""
    mesh_dir = Path(config.mesh_dir).resolve()
    plates = sorted(p.resolve() for p in mesh_dir.glob(plate_glob) if p.is_file())
    if not plates:
        raise FileNotFoundError(f"No files matched {plate_glob!r} in {mesh_dir}")
    return _run_plates(
        plates,
        config,
        log,
        extra_meta={"plate_glob": plate_glob},
    )
