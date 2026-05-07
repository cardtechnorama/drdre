#!/usr/bin/env python
"""CLI: run hemijaw FEA for selected candidate IDs, optionally merge+rank against summary.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent
if str(_PKG_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR.parent))

from fea_ranking_pipeline._cli_common import (
    add_bone_args,
    add_cache_args,
    add_pipeline_args,
    pipeline_kwargs_from_args,
)
from fea_ranking_pipeline.fea_runner import FEARunConfig, run_selected
from fea_ranking_pipeline.merging import join_summary_with_fea, load_summary_candidates
from fea_ranking_pipeline.ranking import compute_rankings
from fea_ranking_pipeline.reporting import log_top_candidates, write_excel


def _parse_ids(spec: str) -> list[int]:
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Run hemijaw FEA for selected candidate IDs and rank them against summary.json",
    )
    ap.add_argument("--ids", type=_parse_ids, required=True, help="Comma-separated candidate IDs, e.g. 13,1,2,25,29")
    ap.add_argument("--mesh-dir", type=Path, required=True, help="Folder with candidate_*_straight.obj")
    ap.add_argument("--out-root", type=Path, required=True, help="Folder for per-candidate subfolders and combined JSON")
    ap.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional summary.json for RMSE/color metrics; enables ranking output",
    )
    ap.add_argument("--summary-out", type=Path, default=None, help="Override combined FEA JSON path")
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--excel", action="store_true", help="Also export an xlsx with the ranked table")
    ap.add_argument("--excel-path", type=Path, default=None)
    add_bone_args(ap)
    add_cache_args(ap)
    add_pipeline_args(ap)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    config = FEARunConfig(
        bone_txt=args.bone_txt.resolve(),
        bone_obj=args.bone_obj.resolve(),
        mesh_dir=args.mesh_dir.resolve(),
        out_root=args.out_root.resolve(),
        mechanical_fem_dir=args.mechanical_fem_dir.resolve(),
        fea_mesh_cache=args.fea_mesh_cache.resolve() if args.fea_mesh_cache else None,
        rebuild_cache=bool(args.rebuild_cache),
        stop_on_error=bool(args.stop_on_error),
        pipeline_kwargs=pipeline_kwargs_from_args(args),
    )

    payload = run_selected(args.ids, config)
    summary_out = args.summary_out.resolve() if args.summary_out else (config.out_root / "fea_results_selected.json")
    summary_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote FEA summary: {summary_out}")
    print(f"FEA: success={payload['n_success']} failed={payload['n_failed']} / total={payload['n_total']}")
    for f in payload["failures"]:
        print(f"  FAIL {f['candidate']}: {f.get('error', 'unknown error')}")

    if args.summary_json is None:
        return 0 if payload["n_failed"] == 0 else 1

    summary_cands = load_summary_candidates(args.summary_json.resolve())
    metrics_by_id = {
        r["candidate_id"]: (r.get("result", {}).get("metrics", {}) or {})
        for r in payload["results"]
        if "candidate_id" in r
    }
    df = join_summary_with_fea(summary_cands, metrics_by_id, selected_ids=args.ids)
    if df.empty:
        print("Merge produced no rows; nothing to rank.")
        return 0 if payload["n_failed"] == 0 else 1

    ranked = compute_rankings(df)
    log_top_candidates(ranked, top_n=args.top_n)

    if args.excel:
        out_xlsx = args.excel_path.resolve() if args.excel_path else (config.out_root / "ranked.xlsx")
        write_excel(ranked, out_xlsx)
        print(f"Wrote Excel: {out_xlsx}")

    return 0 if payload["n_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
