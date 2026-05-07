"""Central configuration for the Interference Streamlit app.

All paths are resolved relative to the repository root by default so the
application stays portable across machines while remaining deterministic.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
APP_ROOT: Path = Path(__file__).resolve().parent

DEFAULT_RUNS_ROOT: Path = APP_ROOT / "runs"
DEFAULT_CONFIG_PATH: Path = APP_ROOT / "configs" / "default_local.json"

STAGE_SEGMENTATION: str = "segmentation"
STAGE_RECONSTRUCTION: str = "reconstruction"
STAGE_MINIPLATE: str = "miniplate"
STAGE_FEA: str = "fea"
STAGE_RANKING: str = "ranking"

STAGE_ORDER: tuple[str, ...] = (
    STAGE_SEGMENTATION,
    STAGE_RECONSTRUCTION,
    STAGE_MINIPLATE,
    STAGE_FEA,
    STAGE_RANKING,
)


DEFAULT_RANDLANET_CKPT: Path = REPO_ROOT / "RandLA-Net" / "best_3cls.pt"
DEFAULT_RANDLANET_RUNNER: Path = APP_ROOT / "bin" / "run_segmentation_randlanet.py"


@dataclass(frozen=True)
class SegmentationModelConfig:
    """Configuration for running the trained segmentation model.

    Currently supports ``backend='randlanet'`` which invokes the RandLA-Net
    inference runner shipped with this app. The runner writes a JSON TXT
    with predicted per-point class colors and copies the input OBJ through.

    ``keep_gray`` controls whether TXT points labeled Gray are dropped before
    inference (``False`` matches upstream ``inference.py``; ``True`` keeps full
    clouds typical of clinical exports).
    """

    backend: str = "randlanet"
    checkpoint: Path = DEFAULT_RANDLANET_CKPT
    runner_script: Path = DEFAULT_RANDLANET_RUNNER
    classes: int = 3
    gpu: int = -1
    keep_gray: bool = True


@dataclass(frozen=True)
class SegmentationConfig:
    """Configuration for the segmentation stage adapter.

    Three modes are supported, resolved in priority order:
      1. ``cache_root``: per-case segmented TXT/OBJ are copied in if present.
      2. ``model``: invokes the configured inference backend (default RandLA-Net).
      3. ``command``: a shell command template executed with ``{in_txt}``,
         ``{in_obj}``, ``{out_dir}``, and ``{case_id}`` substituted.
    """

    cache_root: Path | None = None
    model: SegmentationModelConfig | None = field(default_factory=SegmentationModelConfig)
    command: str | None = None
    timeout_sec: int = 1800


@dataclass(frozen=True)
class ReconstructionConfig:
    """Configuration for the reconstruction adapter."""

    script_path: Path = REPO_ROOT / "reconstruction" / "reconstruction.py"
    split_k: int = 24
    split_min_size: int = 2000
    split_thr_list: str = (
        "1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.35,0.3,0.25,0.2,"
        "0.18,0.15,0.12,0.1,0.08,0.06,0.05"
    )
    icp_iters: int = 50
    d_min: float = 0.5
    backoff_steps: int = 200
    timeout_sec: int = 3600


@dataclass(frozen=True)
class MiniplateCacheConfig:
    """Configuration for the miniplate cached-output loader.

    ``cache_root`` must contain per-case subfolders matching the user-provided
    ``case_id`` (or a ``default`` folder as fallback). Each folder must hold:
      - ``summary.json`` (candidates summary as produced by the miniplate CLI)
      - ``meshes/candidate_XXXX_straight.obj`` plate meshes

    Optional extras: ``fea_results.json``, HTML visualizations, ``candidate_*.txt``.
    """

    cache_root: Path = REPO_ROOT / "miniplate_screws_pipeline" / "_cache"
    fallback_case: str = "default"


@dataclass(frozen=True)
class FEAConfig:
    """Configuration for the FEA / ranking adapter.

    FEA execution itself lives in an external ``mechanical_fem_pipeline``.
    The adapter prefers cached ``fea_results.json`` shipped with the miniplate
    cache when available; otherwise it will invoke the batch FEA runner.
    """

    use_cached_results: bool = True
    plate_glob: str = "candidate_*_straight.obj"
    top_n: int = 10
    export_excel: bool = True


@dataclass(frozen=True)
class AppConfig:
    runs_root: Path = DEFAULT_RUNS_ROOT
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    miniplate: MiniplateCacheConfig = field(default_factory=MiniplateCacheConfig)
    fea: FEAConfig = field(default_factory=FEAConfig)

    def to_dict(self) -> dict[str, Any]:
        seg_model = self.segmentation.model
        return {
            "runs_root": str(self.runs_root),
            "segmentation": {
                "cache_root": str(self.segmentation.cache_root)
                if self.segmentation.cache_root
                else None,
                "model": None
                if seg_model is None
                else {
                    "backend": seg_model.backend,
                    "checkpoint": str(seg_model.checkpoint),
                    "runner_script": str(seg_model.runner_script),
                    "classes": seg_model.classes,
                    "gpu": seg_model.gpu,
                    "keep_gray": seg_model.keep_gray,
                },
                "command": self.segmentation.command,
                "timeout_sec": self.segmentation.timeout_sec,
            },
            "reconstruction": {
                "script_path": str(self.reconstruction.script_path),
                "split_k": self.reconstruction.split_k,
                "split_min_size": self.reconstruction.split_min_size,
                "split_thr_list": self.reconstruction.split_thr_list,
                "icp_iters": self.reconstruction.icp_iters,
                "d_min": self.reconstruction.d_min,
                "backoff_steps": self.reconstruction.backoff_steps,
                "timeout_sec": self.reconstruction.timeout_sec,
            },
            "miniplate": {
                "cache_root": str(self.miniplate.cache_root),
                "fallback_case": self.miniplate.fallback_case,
            },
            "fea": {
                "use_cached_results": self.fea.use_cached_results,
                "plate_glob": self.fea.plate_glob,
                "top_n": self.fea.top_n,
                "export_excel": self.fea.export_excel,
            },
        }


def _as_path(value: Any) -> Path | None:
    if value is None or value == "":
        return None
    return Path(str(value)).expanduser()


def load_app_config(path: Path | None = None) -> AppConfig:
    """Load an :class:`AppConfig` from a JSON file, falling back to defaults."""
    cfg_path = path or Path(os.environ.get("INTERFERENCE_APP_CONFIG", DEFAULT_CONFIG_PATH))
    data: Mapping[str, Any] = {}
    if cfg_path.is_file():
        data = json.loads(cfg_path.read_text(encoding="utf-8"))

    seg_data = dict(data.get("segmentation") or {})
    rec_data = dict(data.get("reconstruction") or {})
    mini_data = dict(data.get("miniplate") or {})
    fea_data = dict(data.get("fea") or {})

    runs_root = _as_path(data.get("runs_root")) or DEFAULT_RUNS_ROOT

    model_data = seg_data.get("model")
    if model_data is None:
        seg_model = SegmentationModelConfig()
    elif model_data is False:
        seg_model = None
    else:
        md = dict(model_data)
        seg_model = SegmentationModelConfig(
            backend=str(md.get("backend", "randlanet")),
            checkpoint=_as_path(md.get("checkpoint")) or DEFAULT_RANDLANET_CKPT,
            runner_script=_as_path(md.get("runner_script")) or DEFAULT_RANDLANET_RUNNER,
            classes=int(md.get("classes", 3)),
            gpu=int(md.get("gpu", -1)),
            keep_gray=bool(md.get("keep_gray", True)),
        )
    seg = SegmentationConfig(
        cache_root=_as_path(seg_data.get("cache_root")),
        model=seg_model,
        command=seg_data.get("command"),
        timeout_sec=int(seg_data.get("timeout_sec", 1800)),
    )
    rec = ReconstructionConfig(
        script_path=_as_path(rec_data.get("script_path"))
        or ReconstructionConfig.__dataclass_fields__["script_path"].default,
        split_k=int(rec_data.get("split_k", 24)),
        split_min_size=int(rec_data.get("split_min_size", 2000)),
        split_thr_list=str(
            rec_data.get(
                "split_thr_list",
                "1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.35,0.3,0.25,0.2,"
                "0.18,0.15,0.12,0.1,0.08,0.06,0.05",
            )
        ),
        icp_iters=int(rec_data.get("icp_iters", 50)),
        d_min=float(rec_data.get("d_min", 2.0)),
        backoff_steps=int(rec_data.get("backoff_steps", 200)),
        timeout_sec=int(rec_data.get("timeout_sec", 3600)),
    )
    mini = MiniplateCacheConfig(
        cache_root=_as_path(mini_data.get("cache_root"))
        or MiniplateCacheConfig.__dataclass_fields__["cache_root"].default,
        fallback_case=str(mini_data.get("fallback_case", "default")),
    )
    fea = FEAConfig(
        use_cached_results=bool(fea_data.get("use_cached_results", True)),
        plate_glob=str(fea_data.get("plate_glob", "candidate_*_straight.obj")),
        top_n=int(fea_data.get("top_n", 10)),
        export_excel=bool(fea_data.get("export_excel", True)),
    )
    return AppConfig(
        runs_root=runs_root,
        segmentation=seg,
        reconstruction=rec,
        miniplate=mini,
        fea=fea,
    )
