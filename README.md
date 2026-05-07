# Interference Viewer (Precomputed)

This bundle is a **Streamlit-ready, precomputed viewer**.
It contains only:
- viewer code used by the app
- interference_app/input_data/
- interference_app/output_data/

It does **not** run segmentation/reconstruction/miniplate/FEA pipelines.

## Run

```bash
pip install -r interference_app/requirements.txt
streamlit run interference_app/ui/streamlit_app.py
```

## Optional env overrides

- INTERFERENCE_VIEWER_INPUT_ROOT
- INTERFERENCE_VIEWER_SEGMENTATION_ROOT
- INTERFERENCE_VIEWER_RECONSTRUCTION_ROOT
- INTERFERENCE_VIEWER_MINIPLATE_ROOT
- INTERFERENCE_VIEWER_FEA_ROOT

## Notes

- The app reads existing artifacts under `interference_app/output_data/`.
- Miniplate candidates are bundled under `interference_app/output_data/miniplate_positions/` and used by default.
- `fea_ranking_pipeline` is included so ranking/merge views import correctly.
- No hardcoded external data path is required; this repo is self-contained.
- Recompute commands are intentionally omitted from this bundle.
