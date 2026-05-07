#!/usr/bin/env python
"""CLI: merge an existing FEA results JSON with a summary.json, rank, and log top-N."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent
if str(_PKG_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR.parent))

from fea_ranking_pipeline.merging import join_summary_with_fea, load_fea_results, load_summary_candidates
from fea_ranking_pipeline.ranking import compute_rankings
from fea_ranking_pipeline.reporting import log_top_candidates, write_excel


def _parse_ids(spec: str | None) -> list[int] | None:
    if not spec:
        return None
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Merge FEA results with summary.json and rank candidates")
    ap.add_argument("--summary-json", type=Path, required=True)
    ap.add_argument(
        "--fea-results",
        type=Path,
        required=True,
        help="Combined FEA JSON (fea_results_all.json or fea_results_selected.json)",
    )
    ap.add_argument("--ids", type=_parse_ids, default=None, help="Optional comma-separated subset of candidate IDs")
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--excel", action="store_true")
    ap.add_argument("--excel-path", type=Path, default=None)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    summary_cands = load_summary_candidates(args.summary_json.resolve())
    metrics_by_id = load_fea_results(args.fea_results.resolve())
    df = join_summary_with_fea(summary_cands, metrics_by_id, selected_ids=args.ids)
    if df.empty:
        print("Merge produced no rows; nothing to rank.")
        return 1

    ranked = compute_rankings(df)
    log_top_candidates(ranked, top_n=args.top_n)

    if args.excel:
        out_xlsx = (
            args.excel_path.resolve()
            if args.excel_path
            else (args.summary_json.resolve().parent / "summary_with_fea_metrics_ranked.xlsx")
        )
        write_excel(ranked, out_xlsx)
        print(f"Wrote Excel: {out_xlsx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
