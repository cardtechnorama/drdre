"""Streamlit entry point: precomputed inference viewer only.

Run locally with:

    streamlit run interference_app/ui/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = APP_DIR.parent
for _p in (str(REPO_ROOT), str(APP_DIR.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st  # noqa: E402

from interference_app.ui.components import render_precomputed_viewer_page  # noqa: E402

st.set_page_config(
    page_title="Interference Viewer",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def main() -> None:
    st.title("Interference Viewer")
    render_precomputed_viewer_page()


if __name__ == "__main__":
    main()
