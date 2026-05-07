# fea_ranking_pipeline

Run FEA on candidates and rank results.

## Modules
- `fea_runner.py` – run FEA (batch / selected)
- `merging.py` – merge inputs
- `ranking.py` – compute ranks
- `reporting.py` – print / export results

## Ranking
- Lower is better: RMSE, FEA metrics  
- Higher is better: Safe% + Caution%  
- Final rank = mean(FEA rank, RMSE rank, color rank)

## CLI

### Batch (all obj files)
```bash
python fea_ranking_pipeline/run_batch_fea.py \
  --mesh-dir <meshes> \
  --out-root <out> \
  --summary-json <summary.json>

### Selected IDs 
python fea_ranking_pipeline/run_selected_fea.py \
  --ids 1,2,3 \
  --mesh-dir <meshes> \
  --out-root <out>

### Ranking (optional Excel export)
```bash
python fea_ranking_pipeline/rank_summary.py \
  --summary-json <summary.json> \
  --fea-results <fea_results.json> \
  --excel
