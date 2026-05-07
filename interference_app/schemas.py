"""Stage input/output schemas for the Interference Streamlit app.

Contracts are declared as frozen dataclasses so they act as stable
interfaces between the UI, orchestrator, and adapters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SKIPPED = "skipped"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass(frozen=True)
class CaseInputs:
    """User-provided inputs for a single case."""

    case_id: str
    input_txt: Path
    input_obj: Path


@dataclass(frozen=True)
class RunRequest:
    """A frozen description of a single end-to-end run."""

    case: CaseInputs
    config_hash: str
    run_id: str
    run_dir: Path


@dataclass(frozen=True)
class SegmentationOutput:
    segmented_txt: Path
    segmented_obj: Path
    source: str


@dataclass(frozen=True)
class ReconstructionOutput:
    reconstructed_txt: Path
    reconstructed_obj: Path


@dataclass(frozen=True)
class MiniplateCacheOutput:
    summary_json: Path
    meshes_dir: Path
    candidate_count: int
    fea_results_json: Path | None


@dataclass(frozen=True)
class FEAOutput:
    fea_results_json: Path
    n_total: int
    n_success: int
    n_failed: int
    from_cache: bool


@dataclass(frozen=True)
class RankingOutput:
    ranked_parquet: Path | None
    ranked_csv: Path
    ranked_xlsx: Path | None
    top_n: int


@dataclass
class StageResult:
    stage: str
    status: StageStatus
    started_at: str
    finished_at: str
    duration_sec: float
    message: str = ""
    artifacts: dict[str, str] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "status": self.status.value,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_sec": round(self.duration_sec, 4),
            "message": self.message,
            "artifacts": dict(self.artifacts),
            "data": dict(self.data),
        }


@dataclass
class RunSummary:
    run_id: str
    case_id: str
    created_at: str
    config_hash: str
    stages: list[StageResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "case_id": self.case_id,
            "created_at": self.created_at,
            "config_hash": self.config_hash,
            "stages": [s.to_dict() for s in self.stages],
        }


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
