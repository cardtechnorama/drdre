"""Page-level Streamlit components for the Interference pipeline app."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as st_components

from ..config import AppConfig, STAGE_ORDER
from ..manifests import build_run_request
from ..pv_streamlit import (
    pyvista_stack_available,
    streamlit_show_bone_and_aimers,
    streamlit_show_bone_and_plate,
    streamlit_show_colored_bone,
)
from ..schemas import CaseInputs, RunSummary, StageResult, StageStatus
from ..visualizations import (
    AIMER_A_COLOR,
    AIMER_B_COLOR,
    BONE_COLOR,
    Mesh,
    PLATE_PALETTE,
    combined_figure,
    combined_figure_bone_and_aimers,
    combined_figure_colored_bone_mesh_txt,
    combined_figure_colored_bone_txt,
    load_obj_mesh,
    load_point_cloud_json,
    mesh_figure,
    mesh_figure_colored_by_txt,
    mesh_negate_x,
    point_cloud_figure,
    transfer_txt_colors_to_mesh_vertices,
    _hex_to_rgb,
)
from ..viz_service import (
    build_miniplate_overview_figure,
    build_miniplate_selection_figure,
    build_reconstruction_figure,
    build_segmentation_figure,
    build_top_n_figure,
    list_candidate_meshes,
    summary_candidate_ids,
)

_STATUS_ICONS: dict[StageStatus, str] = {
    StageStatus.PENDING: "⋯",
    StageStatus.RUNNING: "•",
    StageStatus.SKIPPED: "-",
    StageStatus.SUCCESS: "OK",
    StageStatus.FAILED: "X",
}

_CANDIDATE_ID_RE = re.compile(r"candidate_(\d+)", re.IGNORECASE)

_APP_ROOT = Path(__file__).resolve().parent.parent

# Viewer-only dashboard: fixed defaults (no extra widgets).
_VIEWER_RECON_MESH_MODE = "Mesh — nearest TXT color"
_VIEWER_LAPLACIAN_ITERS = 8


def _viewer_section_columns(section_n: int) -> tuple[Any, Any]:
    """Return ``(col_text, col_viz)`` for a 4-part grid.

    Odd sections: narrow text (1) | wide visualization (3).
    Even sections: wide visualization (3) | narrow text (1).
    Always use ``with col_text:`` then ``with col_viz:`` so copy stays on the "text" side.
    """
    if section_n % 2 == 1:
        c_text, c_viz = st.columns([1, 3])
        return c_text, c_viz
    c_viz, c_text = st.columns([3, 1])
    return c_text, c_viz
def _cached_bone_mesh_vertex_rgb(
    obj_path_str: str,
    txt_path_str: str,
    mode: str,
    k_smooth: int,
    grey_vertices_not_in_txt_ids: bool,
) -> np.ndarray:
    mesh = load_obj_mesh(Path(obj_path_str))
    pc = load_point_cloud_json(Path(txt_path_str), max_points=None)
    return transfer_txt_colors_to_mesh_vertices(
        mesh,
        pc,
        mode=mode,
        k_smooth=k_smooth,
        grey_vertices_not_in_txt_ids=grey_vertices_not_in_txt_ids,
    )


def _default_miniplate_root() -> Path:
    env = os.environ.get("INTERFERENCE_VIEWER_MINIPLATE_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    bundled = (_APP_ROOT / "output_data" / "miniplate_positions").resolve()
    return bundled


def _default_data_roots() -> tuple[Path, Path, Path, Path]:
    input_root = Path(
        os.environ.get(
            "INTERFERENCE_VIEWER_INPUT_ROOT", "interference_app/input_data"
        )
    ).resolve()
    segmentation_root = Path(
        os.environ.get(
            "INTERFERENCE_VIEWER_SEGMENTATION_ROOT",
            "interference_app/output_data/segmentation",
        )
    ).resolve()
    reconstruction_root = Path(
        os.environ.get(
            "INTERFERENCE_VIEWER_RECONSTRUCTION_ROOT",
            "interference_app/output_data/reconstruction",
        )
    ).resolve()
    miniplate_root = _default_miniplate_root()
    return input_root, segmentation_root, reconstruction_root, miniplate_root


def _default_fea_batch_root() -> Path | None:
    raw = os.environ.get("INTERFERENCE_VIEWER_FEA_ROOT", "").strip()
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def _default_aimers_root() -> Path:
    raw = os.environ.get("INTERFERENCE_VIEWER_AIMERS_ROOT", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (_APP_ROOT / "output_data" / "aimers").resolve()


def _resolve_case_aimer_objs(case_id: str) -> tuple[Path | None, Path | None]:
    """Per-case aimers: ``<aimers_root>/<case_id>/aimer_A.obj`` and ``aimer_B.obj``."""
    case_dir = _default_aimers_root() / case_id
    a = case_dir / "aimer_A.obj"
    b = case_dir / "aimer_B.obj"
    if a.is_file() and b.is_file():
        return a.resolve(), b.resolve()
    return None, None


@st.cache_data(show_spinner=False)
def _load_aimer_meshes_cached(aimer_a_str: str, aimer_b_str: str) -> tuple[Mesh, Mesh]:
    return load_obj_mesh(Path(aimer_a_str)), load_obj_mesh(Path(aimer_b_str))


def _render_aimer_prototype_overlay(
    selected_case: str,
    bone_txt: Path,
    aimer_a_path: Path,
    aimer_b_path: Path,
) -> None:
    """Bone colors from input TXT points (per-point Color), plus aimer meshes."""
    pc_bone = load_point_cloud_json(bone_txt, max_points=None)
    if pc_bone.xyz.shape[0] == 0:
        st.caption(f"Input TXT has no points: `{bone_txt}`")
        return

    mesh_a, mesh_b = _load_aimer_meshes_cached(str(aimer_a_path), str(aimer_b_path))
    mesh_a = mesh_negate_x(mesh_a)
    mesh_b = mesh_negate_x(mesh_b)
    title = f"{selected_case}: input bone + surgical aimers"

    fig = combined_figure_colored_bone_txt(
        pc_bone,
        {1: mesh_a, 2: mesh_b},
        [1, 2],
        title=title,
        bone_path=bone_txt,
        bone_marker_size=1.15,
        bone_opacity=0.5,
        plate_opacity=0.92,
        max_bone_points=None,
        plate_names={1: "aimer A", 2: "aimer B"},
        plate_colors={1: AIMER_A_COLOR, 2: AIMER_B_COLOR},
    )
    st.plotly_chart(fig, use_container_width=True, theme=None)


def _discover_fea_results_paths(batch_root: Path) -> list[tuple[str, Path]]:
    """Each case folder contains ``fea_results.json`` (batch FEA CLI output)."""
    if not batch_root.is_dir():
        return []
    out: list[tuple[str, Path]] = []
    for child in sorted(batch_root.iterdir()):
        if not child.is_dir():
            continue
        js = child / "fea_results.json"
        if js.is_file():
            out.append((child.name, js))
    return out


def _flatten_fea_payload(case_id: str, payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ok_rows: list[dict[str, Any]] = []
    for entry in payload.get("results") or []:
        cid = entry.get("candidate_id")
        if cid is None:
            m = _CANDIDATE_ID_RE.search(str(entry.get("candidate", "")))
            cid = int(m.group(1)) if m else None
        metrics = ((entry.get("result") or {}).get("metrics") or {})
        row: dict[str, Any] = {
            "case_id": case_id,
            "candidate_id": cid,
            "candidate": entry.get("candidate"),
            "plate_obj": entry.get("plate_obj"),
        }
        for k, v in metrics.items():
            row[f"fea_{k}"] = v
        ok_rows.append(row)
    fail_rows: list[dict[str, Any]] = []
    for entry in payload.get("failures") or []:
        row = {**entry, "case_id": case_id}
        fail_rows.append(row)
    return ok_rows, fail_rows


@st.cache_data(show_spinner=True)
def _load_fea_batch_tables(batch_root_str: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    batch_root = Path(batch_root_str)
    paths = _discover_fea_results_paths(batch_root)
    if not paths:
        return pd.DataFrame(), pd.DataFrame(), []

    ok_all: list[dict[str, Any]] = []
    fail_all: list[dict[str, Any]] = []
    loaded_cases: list[str] = []

    for case_id, js_path in paths:
        raw = _load_json(str(js_path))
        if raw is None:
            continue
        loaded_cases.append(case_id)
        ok, fail = _flatten_fea_payload(case_id, raw)
        ok_all.extend(ok)
        fail_all.extend(fail)

    df_ok = pd.DataFrame(ok_all)
    df_fail = pd.DataFrame(fail_all)
    return df_ok, df_fail, loaded_cases


def _render_fea_batch_section(batch_root: Path) -> None:
    st.markdown("### FEA batch results (all cases)")
    if not batch_root.is_dir():
        st.info(f"FEA batch root is not a directory: `{batch_root}`")
        return

    df_ok, df_fail, cases = _load_fea_batch_tables(str(batch_root.resolve()))
    if not cases:
        st.warning(
            f"No ``fea_results.json`` found under `{batch_root}` "
            "(expected `<batch_root>/<case_id>/fea_results.json`)."
        )
        return

    st.caption(f"Loaded **{len(cases)}** case folder(s) with FEA JSON.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Cases with FEA JSON", len(cases))
    with c2:
        st.metric("Successful plate runs", len(df_ok))
    with c3:
        st.metric("Failed plate runs", len(df_fail))

    filter_cases = st.multiselect(
        "Filter by case",
        options=cases,
        default=cases,
    )
    if filter_cases:
        show_ok = df_ok[df_ok["case_id"].isin(filter_cases)] if not df_ok.empty else df_ok
        show_fail = df_fail[df_fail["case_id"].isin(filter_cases)] if not df_fail.empty else df_fail
    else:
        show_ok, show_fail = df_ok, df_fail

    if not show_ok.empty:
        st.markdown("**FEA metrics (success)**")
        st.dataframe(show_ok, hide_index=True, use_container_width=True)
        csv_ok = show_ok.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download combined FEA table (CSV)",
            data=csv_ok,
            file_name="fea_batch_success.csv",
            mime="text/csv",
        )
    else:
        st.info("No successful FEA rows in the filtered selection.")

    if not show_fail.empty:
        with st.expander("Failures", expanded=False):
            st.dataframe(show_fail, hide_index=True, use_container_width=True)
            csv_fail = show_fail.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download failures (CSV)",
                data=csv_fail,
                file_name="fea_batch_failures.csv",
                mime="text/csv",
            )


def _save_upload_to_disk(uploaded_file: Any, dest_dir: Path, suffix: str) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"upload{suffix}"
    with dest.open("wb") as f:
        f.write(uploaded_file.getbuffer())
    return dest


def render_run_page(app_config: AppConfig) -> CaseInputs | None:
    st.subheader("Start a run")

    with st.form("run_form", clear_on_submit=False):
        case_id = st.text_input(
            "Case ID",
            value="",
            help="Unique identifier for this case. Used as the run folder prefix "
            "and to locate cached miniplate outputs.",
        )

        col1, col2 = st.columns(2)
        with col1:
            txt_upload = st.file_uploader(
                "Input TXT (colored JSON point cloud)",
                type=["txt", "json"],
                accept_multiple_files=False,
                key="upload_txt",
            )
            txt_path_str = st.text_input(
                "…or path to existing TXT",
                value="",
                help="Use this instead of uploading to reference an on-disk file.",
            )
        with col2:
            obj_upload = st.file_uploader(
                "Input OBJ (mesh)",
                type=["obj"],
                accept_multiple_files=False,
                key="upload_obj",
            )
            obj_path_str = st.text_input(
                "…or path to existing OBJ",
                value="",
            )

        submitted = st.form_submit_button("Run pipeline", type="primary")

    if not submitted:
        return None

    if not case_id.strip():
        st.error("Case ID is required.")
        return None

    request_preview = build_run_request(
        CaseInputs(
            case_id=case_id.strip(),
            input_txt=Path(txt_path_str or "placeholder.txt"),
            input_obj=Path(obj_path_str or "placeholder.obj"),
        ),
        app_config,
    )
    uploads_dir = request_preview.run_dir / "uploads"

    input_txt: Path | None = None
    if txt_upload is not None:
        input_txt = _save_upload_to_disk(txt_upload, uploads_dir, ".txt")
    elif txt_path_str.strip():
        input_txt = Path(txt_path_str).expanduser()

    input_obj: Path | None = None
    if obj_upload is not None:
        input_obj = _save_upload_to_disk(obj_upload, uploads_dir, ".obj")
    elif obj_path_str.strip():
        input_obj = Path(obj_path_str).expanduser()

    if input_txt is None or input_obj is None:
        st.error("Provide both a TXT and an OBJ input, either by upload or path.")
        return None
    if not input_txt.is_file():
        st.error(f"TXT not found: {input_txt}")
        return None
    if not input_obj.is_file():
        st.error(f"OBJ not found: {input_obj}")
        return None

    return CaseInputs(case_id=case_id.strip(), input_txt=input_txt, input_obj=input_obj)


def _stage_row(stage: StageResult) -> dict[str, Any]:
    return {
        "Stage": stage.stage,
        "Status": f"{_STATUS_ICONS.get(stage.status, '?')} {stage.status.value}",
        "Duration (s)": round(stage.duration_sec, 3),
        "Message": stage.message,
    }


def render_progress_page(summary: RunSummary | None, run_dir: Path | None) -> None:
    st.subheader("Progress")
    if summary is None or run_dir is None:
        st.info("No active run. Start one from the Run page.")
        return

    st.caption(f"Run ID: `{summary.run_id}`  ·  Case: `{summary.case_id}`")
    st.caption(f"Run folder: `{run_dir}`")

    rows = [_stage_row(s) for s in summary.stages]
    df = pd.DataFrame(rows, columns=["Stage", "Status", "Duration (s)", "Message"])
    st.dataframe(df, hide_index=True, use_container_width=True)

    for stage in summary.stages:
        if not stage.artifacts and not stage.data:
            continue
        with st.expander(f"{stage.stage}  ·  artifacts & data", expanded=False):
            if stage.artifacts:
                st.markdown("**Artifacts**")
                for key, path in stage.artifacts.items():
                    st.write(f"`{key}` → `{path}`")
            if stage.data:
                st.markdown("**Data**")
                st.json(stage.data)


def _latest_stage(summary: RunSummary, stage_name: str) -> StageResult | None:
    for s in summary.stages:
        if s.stage == stage_name:
            return s
    return None


def _render_ranked_table(csv_path: Path, top_n: int) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to read ranking CSV: {exc}")
        return None
    if df.empty:
        st.warning("Ranking CSV is empty.")
        return df

    st.markdown("**Top candidates**")
    top = df.head(int(top_n) if top_n and top_n > 0 else len(df))
    top_h = int(min(len(top), 25) * 35 + 35)
    st.dataframe(top, hide_index=True, use_container_width=True, height=top_h)

    with st.expander("All ranked candidates", expanded=True):
        full_h = int(min(len(df), 60) * 35 + 35)
        st.dataframe(df, hide_index=True, use_container_width=True, height=full_h)
    return df


def _download_button_for(path: Path, label: str) -> None:
    if not path.is_file():
        return
    with path.open("rb") as f:
        st.download_button(
            label,
            data=f.read(),
            file_name=path.name,
            use_container_width=False,
        )


def _load_manifest_cached(run_dir: Path) -> dict[str, Any] | None:
    from ..manifests import load_manifest

    return load_manifest(run_dir)


def _coerce_candidate_id(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        import re as _re

        m = _re.search(r"(\d+)", str(value))
        if not m:
            return None
        try:
            return int(m.group(1))
        except ValueError:
            return None


def _extract_top_candidate_ids(df: pd.DataFrame, top_k: int) -> list[int]:
    rank_col = next(
        (c for c in ("Rank overall", "rank_overall", "rank") if c in df.columns), None
    )
    id_col = next(
        (c for c in ("Candidate", "candidate_id", "candidate") if c in df.columns), None
    )
    if id_col is None:
        return []
    ordered = df.sort_values(rank_col, ascending=True) if rank_col is not None else df
    ids: list[int] = []
    for val in ordered[id_col].tolist():
        cid = _coerce_candidate_id(val)
        if cid is not None and cid not in ids:
            ids.append(cid)
        if len(ids) >= top_k:
            break
    return ids


@st.cache_data(show_spinner=False)
def _segmentation_fig_cached(run_dir_str: str, cache_key: str):  # noqa: ANN001
    run_dir = Path(run_dir_str)
    manifest = _load_manifest_cached(run_dir)
    if manifest is None:
        return None
    return build_segmentation_figure(run_dir, manifest)


@st.cache_data(show_spinner=False)
def _reconstruction_fig_cached(run_dir_str: str, cache_key: str):  # noqa: ANN001
    run_dir = Path(run_dir_str)
    manifest = _load_manifest_cached(run_dir)
    if manifest is None:
        return None
    return build_reconstruction_figure(run_dir, manifest)


@st.cache_data(show_spinner=False)
def _miniplate_overview_fig_cached(run_dir_str: str, cache_key: str):  # noqa: ANN001
    run_dir = Path(run_dir_str)
    manifest = _load_manifest_cached(run_dir)
    if manifest is None:
        return None
    return build_miniplate_overview_figure(run_dir, manifest)


@st.cache_data(show_spinner=False)
def _miniplate_selection_fig_cached(
    run_dir_str: str, cache_key: str, selected_tuple: tuple[int, ...]
):
    run_dir = Path(run_dir_str)
    manifest = _load_manifest_cached(run_dir)
    if manifest is None:
        return None
    return build_miniplate_selection_figure(manifest, selected_ids=selected_tuple)


@st.cache_data(show_spinner=False)
def _top_n_fig_cached(
    run_dir_str: str, cache_key: str, top_tuple: tuple[int, ...]
):
    run_dir = Path(run_dir_str)
    manifest = _load_manifest_cached(run_dir)
    if manifest is None:
        return None
    return build_top_n_figure(run_dir, manifest, top_tuple)


def _stage_cache_key(summary: RunSummary, stage_name: str) -> str:
    stage = _latest_stage(summary, stage_name)
    if stage is None:
        return stage_name
    return f"{stage_name}:{stage.status.value}:{stage.finished_at}:{len(stage.artifacts)}"


def _render_segmentation_tab(run_dir: Path, summary: RunSummary) -> None:
    stage = _latest_stage(summary, "segmentation")
    if stage is None or stage.status is not StageStatus.SUCCESS:
        st.info("Segmentation has not completed yet.")
        return
    st.caption(stage.message or "Segmentation complete.")
    with st.spinner("Rendering segmented point cloud..."):
        fig = _segmentation_fig_cached(
            str(run_dir), _stage_cache_key(summary, "segmentation")
        )
    if fig is None:
        st.error("Could not load segmentation artifacts for visualization.")
        return
    st.plotly_chart(fig, use_container_width=True, theme=None)
    html_path = Path(stage.artifacts.get("viz_html", ""))
    if html_path.is_file():
        _download_button_for(html_path, "Download segmentation HTML")


def _render_reconstruction_tab(run_dir: Path, summary: RunSummary) -> None:
    stage = _latest_stage(summary, "reconstruction")
    if stage is None or stage.status is not StageStatus.SUCCESS:
        st.info("Reconstruction has not completed yet.")
        return
    st.caption(stage.message or "Reconstruction complete.")
    with st.spinner("Rendering reconstructed mesh..."):
        fig = _reconstruction_fig_cached(
            str(run_dir), _stage_cache_key(summary, "reconstruction")
        )
    if fig is None:
        st.error("Could not load reconstruction artifacts for visualization.")
        return
    st.plotly_chart(fig, use_container_width=True, theme=None)
    html_path = Path(stage.artifacts.get("viz_html", ""))
    if html_path.is_file():
        _download_button_for(html_path, "Download reconstruction HTML")


def _render_candidates_tab(run_dir: Path, summary: RunSummary) -> None:
    stage = _latest_stage(summary, "miniplate")
    if stage is None or stage.status is not StageStatus.SUCCESS:
        st.info("Miniplate stage has not completed yet.")
        return
    manifest = _load_manifest_cached(run_dir)
    if manifest is None:
        st.error("Manifest not found.")
        return

    available_ids = sorted(list_candidate_meshes(manifest).keys())
    if not available_ids:
        st.warning("No candidate meshes found in the miniplate cache.")
        return

    summary_order = summary_candidate_ids(manifest)
    display_ids = [cid for cid in summary_order if cid in available_ids] or available_ids

    default_selection = display_ids[: min(3, len(display_ids))]
    selected = st.multiselect(
        "Select miniplate positions to overlay on the bone",
        options=display_ids,
        default=default_selection,
        format_func=lambda cid: f"candidate_{cid:04d}",
    )

    show_all = st.checkbox(
        "Show overview of all candidates",
        value=False,
        help="Renders every candidate plate on top of the bone mesh.",
    )

    if show_all:
        with st.spinner(f"Rendering {len(display_ids)} candidates..."):
            fig = _miniplate_overview_fig_cached(
                str(run_dir), _stage_cache_key(summary, "miniplate")
            )
    else:
        if not selected:
            st.info("Pick one or more candidates above to render an overlay view.")
            return
        with st.spinner(f"Rendering {len(selected)} selected candidate(s)..."):
            fig = _miniplate_selection_fig_cached(
                str(run_dir),
                _stage_cache_key(summary, "miniplate"),
                tuple(sorted(selected)),
            )

    if fig is None:
        st.error("Could not build candidates visualization.")
        return
    st.plotly_chart(fig, use_container_width=True, theme=None)
    html_path = Path(stage.artifacts.get("viz_html", ""))
    if html_path.is_file():
        _download_button_for(html_path, "Download candidates overview HTML")


def _render_ranking_tab(
    run_dir: Path, summary: RunSummary, app_config: AppConfig
) -> None:
    ranking = _latest_stage(summary, "ranking")
    if ranking is None or ranking.status is not StageStatus.SUCCESS:
        st.warning("Ranking has not completed successfully yet.")
        return
    csv_path = Path(ranking.artifacts.get("ranked_csv", ""))
    if not csv_path.is_file():
        st.error(f"Ranked CSV not found: {csv_path}")
        return

    top_n_cfg = int(ranking.data.get("top_n", app_config.fea.top_n) or app_config.fea.top_n)
    df = _render_ranked_table(csv_path, top_n=top_n_cfg)

    if df is not None and not df.empty:
        top_k = min(5, len(df))
        top_ids = _extract_top_candidate_ids(df, top_k)
        if top_ids:
            st.markdown(f"**Top {len(top_ids)} miniplate positions**")
            with st.spinner("Rendering top positions..."):
                fig = _top_n_fig_cached(
                    str(run_dir),
                    _stage_cache_key(summary, "ranking"),
                    tuple(top_ids),
                )
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True, theme=None)
                ranking_html = run_dir / "ranking" / "viz" / "top_positions.html"
                if ranking_html.is_file():
                    _download_button_for(ranking_html, "Download top-positions HTML")

    st.markdown("**Downloads**")
    cols = st.columns(3)
    with cols[0]:
        _download_button_for(csv_path, "Download ranked CSV")
    with cols[1]:
        xlsx_path = Path(ranking.artifacts.get("ranked_xlsx", ""))
        if xlsx_path.is_file():
            _download_button_for(xlsx_path, "Download ranked XLSX")
    with cols[2]:
        zip_path = _maybe_zip_run(run_dir)
        if zip_path is not None and zip_path.is_file():
            _download_button_for(zip_path, "Download full run (zip)")


def render_results_page(
    summary: RunSummary | None,
    run_dir: Path | None,
    app_config: AppConfig,
) -> None:
    st.subheader("Results")
    if summary is None or run_dir is None:
        st.info("No active run. Start one from the Run page.")
        return

    st.caption(f"Run ID: `{summary.run_id}`  ·  Case: `{summary.case_id}`")

    tab_seg, tab_rec, tab_cand, tab_rank = st.tabs(
        ["Segmentation", "Reconstruction", "Candidates", "Ranking"]
    )
    with tab_seg:
        _render_segmentation_tab(run_dir, summary)
    with tab_rec:
        _render_reconstruction_tab(run_dir, summary)
    with tab_cand:
        _render_candidates_tab(run_dir, summary)
    with tab_rank:
        _render_ranking_tab(run_dir, summary, app_config)


def _maybe_zip_run(run_dir: Path) -> Path | None:
    out_zip = run_dir / f"{run_dir.name}_artifacts"
    archive_file = Path(str(out_zip) + ".zip")
    if archive_file.is_file():
        return archive_file
    tmp = Path(tempfile.mkdtemp(prefix="interf_zip_"))
    try:
        stub = tmp / out_zip.name
        shutil.make_archive(str(stub), "zip", root_dir=str(run_dir))
        produced = Path(str(stub) + ".zip")
        if produced.is_file():
            shutil.move(str(produced), str(archive_file))
            return archive_file
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return None


def _resolve_case_file(root: Path, case_id: str, suffix: str) -> Path | None:
    nested = root / case_id / f"{case_id}{suffix}"
    if nested.is_file():
        return nested
    flat = root / f"{case_id}{suffix}"
    if flat.is_file():
        return flat
    return None


def _discover_case_ids(input_root: Path) -> list[str]:
    if not input_root.is_dir():
        return []
    case_ids: list[str] = []
    for obj_path in sorted(input_root.glob("*.obj")):
        case_id = obj_path.stem
        if (input_root / f"{case_id}.txt").is_file():
            case_ids.append(case_id)
    return case_ids


@st.cache_data(show_spinner=False)
def _load_json(path_str: str) -> dict[str, Any] | None:
    path = Path(path_str)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


@st.cache_data(show_spinner=False)
def _load_html(path_str: str) -> str | None:
    path = Path(path_str)
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None


def _candidate_id_from_name(path: Path) -> int | None:
    match = _CANDIDATE_ID_RE.search(path.name)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _candidate_mesh_map(case_miniplate_dir: Path) -> dict[int, Path]:
    meshes_dir = case_miniplate_dir / "meshes"
    if not meshes_dir.is_dir():
        return {}
    candidates: dict[int, Path] = {}
    for mesh_path in sorted(meshes_dir.glob("candidate_*_straight.obj")):
        cid = _candidate_id_from_name(mesh_path)
        if cid is not None:
            candidates.setdefault(cid, mesh_path)
    if candidates:
        return candidates
    for mesh_path in sorted(meshes_dir.glob("candidate_*.obj")):
        cid = _candidate_id_from_name(mesh_path)
        if cid is not None:
            candidates.setdefault(cid, mesh_path)
    return candidates


_RANK_CAND_HTML_RE = re.compile(
    r"^rank_(\d+)_candidate_(\d+)(?:\.html)?$", re.IGNORECASE
)


def _best_color_html_per_candidate(case_miniplate_dir: Path) -> dict[int, Path]:
    """Pick one ``top10_color_html`` file per candidate (lowest rank # = best in top-10)."""
    html_dir = case_miniplate_dir / "top10_color_html"
    if not html_dir.is_dir():
        return {}
    best: dict[int, tuple[int, Path]] = {}
    for html_path in html_dir.glob("*.html"):
        m = _RANK_CAND_HTML_RE.match(html_path.stem)
        if m:
            rank, cid = int(m.group(1)), int(m.group(2))
            prev = best.get(cid)
            if prev is None or rank < prev[0]:
                best[cid] = (rank, html_path)
            continue
        cid = _candidate_id_from_name(html_path)
        if cid is not None and cid not in best:
            best[cid] = (999, html_path)
    return {cid: t[1] for cid, t in best.items()}


def _ranking_dataframe(summary: dict[str, Any] | None) -> pd.DataFrame:
    if not summary:
        return pd.DataFrame()
    rows = summary.get("all_candidates")
    if not isinstance(rows, list):
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "rank" in df.columns:
        df = df.sort_values("rank", ascending=True)
    return df


def _metrics_by_candidate_from_summary(
    summary_data: dict[str, Any] | None,
) -> dict[int, dict[str, Any]]:
    if not summary_data:
        return {}
    rows = summary_data.get("all_candidates") or summary_data.get("candidates") or []
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        cid = row.get("candidate_id")
        if cid is None:
            continue
        try:
            out[int(cid)] = row
        except (TypeError, ValueError):
            continue
    return out


def _resolve_fea_results_json_for_case(
    case_id: str,
    case_miniplate_dir: Path,
    fea_batch_root: Path | None,
) -> Path | None:
    candidates = [case_miniplate_dir / "fea_results.json"]
    if fea_batch_root:
        candidates.append(Path(fea_batch_root) / case_id / "fea_results.json")
    for p in candidates:
        if p.is_file():
            return p.resolve()
    return None


def _try_combined_geo_fea_ranking(
    summary_geo: dict[str, Any],
    summary_path: Path,
    case_id: str,
    case_miniplate_dir: Path,
    fea_batch_root: Path | None,
) -> tuple[pd.DataFrame | None, str]:
    """Merge summary.json candidates with ``fea_results.json`` (same rules as CLI)."""
    try:
        from fea_ranking_pipeline.merging import join_summary_with_fea, load_fea_results
        from fea_ranking_pipeline.ranking import compute_rankings, subset_merged_ranking_for_display
    except ImportError as exc:
        return None, f"fea_ranking_pipeline not importable ({exc})."

    fea_path = _resolve_fea_results_json_for_case(case_id, case_miniplate_dir, fea_batch_root)
    if fea_path is None:
        return (
            None,
            "No fea_results.json for this case: add **FEA batch root** in the sidebar "
            "(folder containing `<case_id>/fea_results.json`), or copy "
            "`fea_results.json` next to `summary.json`.",
        )

    summary_cands = summary_geo.get("all_candidates") or summary_geo.get("candidates") or []
    if not summary_cands:
        return None, "summary.json has no candidate rows."

    metrics_by_id = load_fea_results(fea_path)
    if not metrics_by_id:
        return None, f"FEA results empty or unreadable: `{fea_path}`"

    merged = join_summary_with_fea(summary_cands, metrics_by_id)
    if merged.empty:
        return (
            None,
            "Geometric summary and FEA share **no candidate IDs** "
            "(check candidate_id alignment between JSON files).",
        )

    ranked = subset_merged_ranking_for_display(compute_rankings(merged))
    note = (
        f"Combined **`{fea_path.name}`** (FEA metrics) with **`{summary_path.name}`** "
        f"(class % / RMSE). Rows sorted by pipeline **Final avg rank** / **Rank overall**."
    )
    return ranked, note


def _candidate_select_label(cid: int, metrics_by_id: dict[int, dict[str, Any]]) -> str:
    row = metrics_by_id.get(cid)
    base = f"candidate_{cid:04d}"
    if not row:
        return base
    parts: list[str] = []
    g = row.get("green_pct")
    r = row.get("rmse_mm")
    if g is not None:
        try:
            parts.append(f"green {float(g):.1f}%")
        except (TypeError, ValueError):
            parts.append(f"green {g}%")
    if r is not None:
        try:
            parts.append(f"RMSE {float(r):.3f} mm")
        except (TypeError, ValueError):
            parts.append(f"RMSE {r} mm")
    return f"{base} — {', '.join(parts)}" if parts else base


def _resolve_case_miniplate_dir(miniplate_root: Path, case_id: str) -> Path:
    direct = miniplate_root
    nested = miniplate_root / case_id

    def _looks_like_case_dir(path: Path) -> bool:
        if not path.is_dir():
            return False
        if (path / "summary.json").is_file():
            return True
        meshes_dir = path / "meshes"
        if not meshes_dir.is_dir():
            return False
        return any(meshes_dir.glob("candidate_*.obj"))

    if _looks_like_case_dir(direct):
        return direct
    if _looks_like_case_dir(nested):
        return nested
    return nested


def render_precomputed_viewer_page() -> None:
    st.markdown("### 1) Input")

    (
        default_input_root,
        default_segmentation_root,
        default_reconstruction_root,
        default_miniplate_root,
    ) = _default_data_roots()

    c1_txt, c1_viz = _viewer_section_columns(1)
    # with c1_txt:
    #     st.markdown("Input")

    input_root = default_input_root
    segmentation_root = default_segmentation_root
    reconstruction_root = default_reconstruction_root
    miniplate_root = default_miniplate_root
    fea_batch_raw = str(_default_fea_batch_root() or "")

    fea_batch_root: Path | None = None
    if fea_batch_raw.strip():
        fea_batch_root = Path(fea_batch_raw.strip()).expanduser()

    if fea_batch_root is not None:
        _render_fea_batch_section(fea_batch_root)

    case_ids = _discover_case_ids(input_root)
    if not case_ids:
        st.error(f"No valid cases found in `{input_root}`.")
        return

    with c1_viz:
        selected_case = st.selectbox(
            "Select one bone case",
            options=case_ids,
            index=0,
            help="Pick one from loaded cases before viewing segmentation/reconstruction/miniplates.",
        )

    input_txt = _resolve_case_file(input_root, selected_case, ".txt")
    input_obj = _resolve_case_file(input_root, selected_case, ".obj")
    seg_txt = _resolve_case_file(segmentation_root, selected_case, ".txt")
    seg_obj = _resolve_case_file(segmentation_root, selected_case, ".obj")
    rec_txt = _resolve_case_file(reconstruction_root, selected_case, ".txt")
    rec_obj = _resolve_case_file(reconstruction_root, selected_case, ".obj")

    # st.markdown("**Input pair for selected case**")
    # cols_in = st.columns(2)
    # with cols_in[0]:
    #     st.write(f"TXT: `{input_txt}`" if input_txt else "TXT: missing")
    # with cols_in[1]:
    #     st.write(f"OBJ: `{input_obj}`" if input_obj else "OBJ: missing")

    st.markdown("### 2) Segmentation")
    c2_txt, c2_viz = _viewer_section_columns(2)
    if seg_txt and seg_obj:
        seg_fig = point_cloud_figure(
            load_point_cloud_json(seg_txt),
            title=f"{selected_case} segmentation",
        )
        with c2_viz:
            st.plotly_chart(seg_fig, use_container_width=True, theme=None)
        with c2_txt:
            st.markdown(
                "Uses the RandLA-Net model to segment regions and assign colors for the point cloud bone."
                ""
            )
    else:
        with c2_viz:
            st.warning("Segmentation artifacts are missing for this case.")
        with c2_txt:
            st.markdown(
                "**Semantic point cloud**\n\n"
                "Expected segmentation **TXT/OBJ** for this case. That cloud is what the "
                "miniplate code turns into **region rules** (allowed vs fracture labels) "
                "via nearest-neighbor color lookup on the bone."
            )

    st.markdown("### 3) Reconstruction")
    has_rec_txt = bool(rec_txt and rec_txt.is_file())
    has_rec_obj = bool(rec_obj and rec_obj.is_file())
    recon_pc_nonempty = False
    if has_rec_txt:
        _probe = load_point_cloud_json(rec_txt, max_points=1)
        recon_pc_nonempty = bool(_probe.xyz.shape[0] > 0)

    pv_ok = pyvista_stack_available()
    use_pyvista_mesh = bool(pv_ok)
    laplacian_iters = _VIEWER_LAPLACIAN_ITERS if pv_ok else 0
    bone_tri_cap: int | None = None
    grey_unlisted = True

    st.session_state["interference_recon_display"] = _VIEWER_RECON_MESH_MODE
    st.session_state["interference_mesh_engine"] = (
        "PyVista (smooth VTK)" if pv_ok else "Plotly"
    )
    st.session_state["pv_laplace"] = laplacian_iters

    recon_display = _VIEWER_RECON_MESH_MODE

    c3_txt, c3_viz = _viewer_section_columns(3)
    with c3_txt:
        st.markdown(
            "Uses purple point regions and aligns bones using the Kabsch and ICP algorithms."
        )
    with c3_viz:
        if has_rec_txt and has_rec_obj and recon_pc_nonempty:
            mesh_rec = load_obj_mesh(rec_obj)
            pc_full = load_point_cloud_json(rec_txt, max_points=None)
            mode = "nearest" if "nearest" in recon_display else "smooth"
            if use_pyvista_mesh and pv_ok:
                v_rgb = _cached_bone_mesh_vertex_rgb(
                    str(rec_obj.resolve()),
                    str(rec_txt.resolve()),
                    mode,
                    12,
                    grey_unlisted,
                )
                ok = streamlit_show_colored_bone(
                    mesh_rec,
                    v_rgb,
                    streamlit_key=f"pv3_{selected_case}_{mode}",
                    mesh_path=rec_obj,
                    laplacian_iters=laplacian_iters,
                    max_faces=bone_tri_cap,
                )
                if not ok:
                    st.plotly_chart(
                        mesh_figure_colored_by_txt(
                            mesh_rec,
                            pc_full,
                            title=(
                                f"{selected_case} reconstructed bone "

                            ),
                            mesh_path=rec_obj,
                            mode=mode,
                            k_smooth=12,
                            grey_vertices_not_in_txt_ids=grey_unlisted,
                            max_faces=bone_tri_cap,
                        ),
                        use_container_width=True,
                        theme=None,
                    )
            else:
                st.plotly_chart(
                    mesh_figure_colored_by_txt(
                        mesh_rec,
                        pc_full,
                        title=(
                            f"{selected_case} reconstructed bone "
                            f"(mesh, {mode} TXT colors)"
                        ),
                        mesh_path=rec_obj,
                        mode=mode,
                        k_smooth=12,
                        grey_vertices_not_in_txt_ids=grey_unlisted,
                        max_faces=bone_tri_cap,
                    ),
                    use_container_width=True,
                    theme=None,
                )
        elif has_rec_txt and recon_pc_nonempty:
            pc_rec = load_point_cloud_json(rec_txt)
            rec_fig = point_cloud_figure(
                pc_rec,
                title=f"{selected_case} reconstructed bone (from TXT)",
            )
            st.plotly_chart(rec_fig, use_container_width=True, theme=None)
        elif has_rec_obj:
            rec_fig = mesh_figure(
                load_obj_mesh(rec_obj),
                title=f"{selected_case} reconstructed bone (from OBJ)",
                mesh_path=rec_obj,
            )
            st.plotly_chart(rec_fig, use_container_width=True, theme=None)
        else:
            st.info("No reconstruction TXT or OBJ found for this case.")

    case_miniplate_dir = _resolve_case_miniplate_dir(miniplate_root, selected_case)
    summary_path_geo = case_miniplate_dir / "summary.json"
    summary_geo = _load_json(str(summary_path_geo))
    metrics_by_candidate = _metrics_by_candidate_from_summary(summary_geo)

    st.markdown("### 4) Miniplate selection")
    if not rec_obj or not rec_obj.is_file():
        st.warning(
            "Miniplate overlay uses reconstruction bone OBJ with plate OBJ. "
            f"Missing: `{reconstruction_root}` / `{selected_case}.obj`"
        )
        return

    candidate_meshes = _candidate_mesh_map(case_miniplate_dir)
    # color_html_by_cand = _best_color_html_per_candidate(case_miniplate_dir)

    if not candidate_meshes:
        st.warning(f"No candidate meshes found in `{case_miniplate_dir / 'meshes'}`.")
        return

    c4_txt, c4_viz = _viewer_section_columns(4)
    with c4_txt:
        st.markdown(
            "Miniplate positions are generated on the surface of the inferior bone.\n\n"
            "RMSE is calculated based on the distance between the holes and the surface.\n\n"
            "The green region percentage is calculated from the screw contact points touching the surface "
            "(based on their color)."
        )

    with c4_viz:
        candidate_ids = sorted(candidate_meshes.keys())
        selected_candidate = st.selectbox(
            "Select one miniplate candidate",
            options=candidate_ids,
            format_func=lambda cid: _candidate_select_label(cid, metrics_by_candidate),
            index=0,
        )

        row_m = metrics_by_candidate.get(selected_candidate)
        mc1, mc2 = st.columns(2)
        with mc1:
            if row_m is not None and row_m.get("green_pct") is not None:
                try:
                    st.metric(
                        "Green region",
                        f"{float(row_m['green_pct']):.2f}%",
                    )
                except (TypeError, ValueError):
                    st.metric("Green region", str(row_m.get("green_pct")))
            else:
                st.caption("Green % — not in summary.json for this candidate.")
        with mc2:
            if row_m is not None and row_m.get("rmse_mm") is not None:
                try:
                    st.metric(
                        "RMSE",
                        f"{float(row_m['rmse_mm']):.3f} mm",
                    )
                except (TypeError, ValueError):
                    st.metric("RMSE", str(row_m.get("rmse_mm")))
            else:
                st.caption("RMSE — not in summary.json for this candidate.")

        selected_plate = candidate_meshes[selected_candidate]
        plate_mesh_obj = load_obj_mesh(selected_plate)
        plot_title_base = (
            f"{selected_case}: candidate_{selected_candidate:04d} on reconstructed bone"
        )
        mesh_rec = load_obj_mesh(rec_obj)
        recon_display = st.session_state.get("interference_recon_display")
        use_mesh_txt_colors = recon_display in (
            "Mesh — nearest TXT color",
            "Mesh — smooth TXT blend",
        )
        bone_is_txt_points = recon_display == "Classified points (TXT)"
        mesh_engine_lbl = st.session_state.get("interference_mesh_engine", "")
        use_pv_overlay = pv_ok and str(mesh_engine_lbl).startswith("PyVista")
        lap_overlay = int(st.session_state.get("pv_laplace", 0)) if use_pv_overlay else 0
        plate_hex = PLATE_PALETTE[selected_candidate % len(PLATE_PALETTE)]

        pv_shown = False
        if (
            use_pv_overlay
            and mesh_rec.faces.shape[0] > 0
            and not bone_is_txt_points
        ):
            if (
                use_mesh_txt_colors
                and rec_txt
                and rec_txt.is_file()
                and load_point_cloud_json(rec_txt, max_points=1).xyz.shape[0] > 0
            ):
                mode = "nearest" if recon_display == "Mesh — nearest TXT color" else "smooth"
                v_rgb_ov = _cached_bone_mesh_vertex_rgb(
                    str(rec_obj.resolve()),
                    str(rec_txt.resolve()),
                    mode,
                    12,
                    grey_unlisted,
                )
            else:
                g = np.asarray(_hex_to_rgb(BONE_COLOR), dtype=np.uint8).reshape(1, 3)
                v_rgb_ov = np.tile(g, (mesh_rec.vertices.shape[0], 1))
            pv_shown = streamlit_show_bone_and_plate(
                mesh_rec,
                v_rgb_ov,
                plate_mesh_obj,
                plate_color_hex=plate_hex,
                streamlit_key=f"pv4_{selected_case}_{selected_candidate}",
                bone_path=rec_obj,
                plate_path=selected_plate,
                laplacian_iters=lap_overlay,
                max_bone_faces=bone_tri_cap,
                max_plate_faces=None,
            )

        if not pv_shown:
            if (
                use_mesh_txt_colors
                and rec_txt
                and rec_txt.is_file()
                and load_point_cloud_json(rec_txt, max_points=1).xyz.shape[0] > 0
            ):
                mode = "nearest" if recon_display == "Mesh — nearest TXT color" else "smooth"
                v_rgb = _cached_bone_mesh_vertex_rgb(
                    str(rec_obj.resolve()),
                    str(rec_txt.resolve()),
                    mode,
                    12,
                    grey_unlisted,
                )
                overlay_fig = combined_figure_colored_bone_mesh_txt(
                    mesh_rec,
                    v_rgb,
                    {selected_candidate: plate_mesh_obj},
                    [selected_candidate],
                    title=f"{plot_title_base}",
                    bone_path=rec_obj,
                    max_bone_faces=bone_tri_cap,
                )
            elif rec_txt and rec_txt.is_file():
                pc_overlay = load_point_cloud_json(rec_txt)
                if pc_overlay.xyz.shape[0] > 0:
                    overlay_fig = combined_figure_colored_bone_txt(
                        pc_overlay,
                        {selected_candidate: plate_mesh_obj},
                        [selected_candidate],
                        title=f"{plot_title_base}",
                        bone_path=rec_txt,
                    )
                else:
                    overlay_fig = combined_figure(
                        bone_mesh=mesh_rec,
                        plate_meshes={selected_candidate: plate_mesh_obj},
                        selected_ids=[selected_candidate],
                        title=f"{plot_title_base} (Plotly — bone mesh, gray)",
                        bone_path=rec_obj,
                    )
            else:
                overlay_fig = combined_figure(
                    bone_mesh=mesh_rec,
                    plate_meshes={selected_candidate: plate_mesh_obj},
                    selected_ids=[selected_candidate],
                    title=f"{plot_title_base} (Plotly — bone mesh, gray)",
                    bone_path=rec_obj,
                )
            st.plotly_chart(overlay_fig, use_container_width=True, theme=None)
    # st.caption(f"Plate OBJ: `{selected_plate}`")

    # html_path = color_html_by_cand.get(selected_candidate)
    # if html_path and html_path.is_file():
    #     html_content = _load_html(str(html_path))
    #     if html_content:
    #         st.markdown(
    #             "**Pipeline HTML (Three.js)** — same coloring idea as Plotly TXT view above."
    #         )
    #         st_components.html(html_content, height=720, scrolling=True)
    #         st.caption(f"HTML: `{html_path}`")
    # else:
    #     st.caption(
    #         "No `top10_color_html` file for this candidate: the exporter only writes "
    #         "snapshots for the **top color-ranked** plates (typically 10). "
    #         "Use Plotly above for full reconstruction TXT colors for any candidate."
    #     )

    st.markdown("### 5) Finite Element Analysis (FEA)")
    if summary_geo is None:
        st.warning(f"Ranking summary missing or invalid: `{summary_path_geo}`")
    else:
        ranked_combo, combo_note = _try_combined_geo_fea_ranking(
            summary_geo,
            summary_path_geo,
            selected_case,
            case_miniplate_dir,
            fea_batch_root,
        )
        c5_txt, c5_viz = _viewer_section_columns(5)
        with c5_txt:
            st.markdown(
                "Finite Element Analysis (FEA) is performed for each miniplate position by applying force (N) "
                "to the middle of the jaw and calculating displacement and other mechanical metrics."
            )
        with c5_viz:
            if ranked_combo is not None and not ranked_combo.empty:
                st.dataframe(ranked_combo, hide_index=True, use_container_width=True)
                cand_col = "Candidate" if "Candidate" in ranked_combo.columns else None
                if cand_col is not None and len(ranked_combo) > 0:
                    top_raw = ranked_combo.iloc[0][cand_col]
                    top_cid = _coerce_candidate_id(top_raw)
                    top_label = (
                        f"candidate_{top_cid:04d}"
                        if top_cid is not None
                        else str(top_raw)
                    )
                    st.success(
                        f"Best candidate by combined geometric + FEA ranking: **{top_label}**."
                    )
            else:
                if combo_note:
                    st.info(combo_note)
                ranking_df = _ranking_dataframe(summary_geo)
                if ranking_df.empty:
                    st.warning("No ranking rows found in summary.")
                else:
                    st.caption(
                        "Showing **geometric summary only** until FEA JSON is available "
                        "for merging."
                    )
                    st.dataframe(ranking_df, hide_index=True, use_container_width=True)
                    best = summary_geo.get("best_by_combined") or {}
                    best_id = best.get("candidate_id")
                    if best_id is not None:
                        st.success(
                            "Best miniplate by geometric summary score: "
                            f"candidate_{int(best_id):04d}"
                        )
                    else:
                        st.info(
                            "`best_by_combined.candidate_id` is not available in summary.json."
                        )

    aimer_a_path, aimer_b_path = _resolve_case_aimer_objs(selected_case)
    has_input_bone = bool(
        input_txt and input_txt.is_file() and input_obj and input_obj.is_file()
    )
    if aimer_a_path is not None and aimer_b_path is not None and has_input_bone:
        st.markdown("### 6) Aimer prototype")
        c6_txt, c6_viz = _viewer_section_columns(6)
        with c6_txt:
            st.markdown(
                "Original **input bone** from `input_data/<case>.txt` (per-point colors), "
                "with aimer shells **A** (red) and **B** (blue). "
                f"Shown for case `{selected_case}` when aimers are uploaded."
            )
        with c6_viz:
            _render_aimer_prototype_overlay(
                selected_case,
                input_txt,
                aimer_a_path,
                aimer_b_path,
            )


__all__ = [
    "render_precomputed_viewer_page",
    "render_run_page",
    "render_progress_page",
    "render_results_page",
    "STAGE_ORDER",
]
