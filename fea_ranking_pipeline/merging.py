"""Load summary candidates and FEA results, then join them into a single DataFrame."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

_CANDIDATE_ID_RE = re.compile(r"candidate[_-]?(\d+)", re.IGNORECASE)


def _parse_candidate_id(name: str) -> int | None:
    m = _CANDIDATE_ID_RE.search(name)
    if not m:
        return None
    return int(m.group(1))


def load_summary_candidates(summary_path: Path) -> list[dict]:
    data = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    cands = data.get("all_candidates") or data.get("candidates") or []
    return list(cands)


def load_fea_results(fea_results_path: Path) -> dict[int, dict]:
    """Return a mapping ``candidate_id -> result.metrics``.

    Supports both formats produced in this repo:
      - ``fea_results_all.json`` entries lacking ``candidate_id`` (parsed from name)
      - ``fea_results_selected.json`` entries with explicit ``candidate_id``
    """
    data = json.loads(Path(fea_results_path).read_text(encoding="utf-8"))
    entries = data.get("results") or []
    out: dict[int, dict] = {}
    for entry in entries:
        cid = entry.get("candidate_id")
        if cid is None:
            cid = _parse_candidate_id(str(entry.get("candidate", "")))
        if cid is None:
            continue
        metrics = ((entry.get("result") or {}).get("metrics") or {})
        out[int(cid)] = dict(metrics)
    return out


def join_summary_with_fea(
    summary_candidates: Iterable[dict],
    metrics_by_id: dict[int, dict],
    selected_ids: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Join summary rows with FEA metrics under a ``fea_<key>`` prefix."""
    by_id: dict[int, dict] = {int(r["candidate_id"]): dict(r) for r in summary_candidates if "candidate_id" in r}
    if selected_ids is None:
        ids = sorted(set(by_id) & set(metrics_by_id)) or sorted(by_id)
    else:
        ids = [int(i) for i in selected_ids]

    rows: list[dict] = []
    for cid in ids:
        base = by_id.get(cid)
        if base is None:
            continue
        row = dict(base)
        for k, v in (metrics_by_id.get(cid) or {}).items():
            row[f"fea_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)
