"""Run identifiers, run folder layout, and manifest helpers.

A run's identity is a deterministic hash of (``case_id``, serialized config).
All stage outputs live beneath ``runs_root/<run_id>/<stage>/``, and a
``manifest.json`` at the run root records stage status, artifact paths,
and parameter/version metadata.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from . import __version__
from .config import APP_ROOT, AppConfig, STAGE_ORDER
from .schemas import (
    CaseInputs,
    RunRequest,
    RunSummary,
    StageResult,
    StageStatus,
    utc_now_iso,
)


def hash_config(config: Mapping[str, Any]) -> str:
    """Return a short, stable hash of a JSON-serializable mapping."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def compute_run_id(case_id: str, config_hash: str) -> str:
    digest = hashlib.sha256(f"{case_id}::{config_hash}".encode("utf-8")).hexdigest()[:12]
    safe_case = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in case_id)
    return f"{safe_case}-{digest}"


def build_run_request(case: CaseInputs, app_config: AppConfig) -> RunRequest:
    cfg_hash = hash_config(app_config.to_dict())
    run_id = compute_run_id(case.case_id, cfg_hash)
    run_dir = app_config.runs_root / run_id
    return RunRequest(case=case, config_hash=cfg_hash, run_id=run_id, run_dir=run_dir)


def stage_dir(run_dir: Path, stage: str) -> Path:
    return run_dir / stage


def ensure_run_layout(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    for stage in STAGE_ORDER:
        stage_dir(run_dir, stage).mkdir(parents=True, exist_ok=True)


def manifest_path(run_dir: Path) -> Path:
    return run_dir / "manifest.json"


def _default_manifest(request: RunRequest) -> dict[str, Any]:
    return {
        "app_version": __version__,
        "run_id": request.run_id,
        "case_id": request.case.case_id,
        "created_at": utc_now_iso(),
        "config_hash": request.config_hash,
        "input_txt": str(request.case.input_txt),
        "input_obj": str(request.case.input_obj),
        "stages": {
            s: {
                "status": StageStatus.PENDING.value,
                "artifacts": {},
                "data": {},
                "message": "",
                "started_at": None,
                "finished_at": None,
                "duration_sec": 0.0,
            }
            for s in STAGE_ORDER
        },
    }


def load_manifest(run_dir: Path) -> dict[str, Any] | None:
    path = manifest_path(run_dir)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(run_dir: Path, manifest: Mapping[str, Any]) -> None:
    manifest_path(run_dir).write_text(
        json.dumps(manifest, indent=2, sort_keys=False), encoding="utf-8"
    )


def initialize_manifest(request: RunRequest, app_config: AppConfig) -> dict[str, Any]:
    """Create or load the manifest for a run, updating volatile fields."""
    existing = load_manifest(request.run_dir)
    manifest = existing if existing is not None else _default_manifest(request)
    manifest["app_version"] = __version__
    manifest["config"] = app_config.to_dict()
    manifest["app_root"] = str(APP_ROOT)
    save_manifest(request.run_dir, manifest)
    return manifest


def record_stage(run_dir: Path, result: StageResult) -> dict[str, Any]:
    manifest = load_manifest(run_dir) or {}
    stages = manifest.setdefault("stages", {})
    stages[result.stage] = result.to_dict()
    save_manifest(run_dir, manifest)
    return manifest


def stage_status(manifest: Mapping[str, Any], stage: str) -> StageStatus:
    info = (manifest.get("stages") or {}).get(stage) or {}
    raw = info.get("status", StageStatus.PENDING.value)
    try:
        return StageStatus(raw)
    except ValueError:
        return StageStatus.PENDING


def summarize_run(manifest: Mapping[str, Any]) -> RunSummary:
    stages: list[StageResult] = []
    for stage_name in STAGE_ORDER:
        info = (manifest.get("stages") or {}).get(stage_name) or {}
        try:
            status = StageStatus(info.get("status", StageStatus.PENDING.value))
        except ValueError:
            status = StageStatus.PENDING
        stages.append(
            StageResult(
                stage=stage_name,
                status=status,
                started_at=info.get("started_at") or "",
                finished_at=info.get("finished_at") or "",
                duration_sec=float(info.get("duration_sec", 0.0) or 0.0),
                message=info.get("message", "") or "",
                artifacts=dict(info.get("artifacts") or {}),
                data=dict(info.get("data") or {}),
            )
        )
    return RunSummary(
        run_id=str(manifest.get("run_id", "")),
        case_id=str(manifest.get("case_id", "")),
        created_at=str(manifest.get("created_at", "")),
        config_hash=str(manifest.get("config_hash", "")),
        stages=stages,
    )
