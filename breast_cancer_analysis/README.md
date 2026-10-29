# Breast Cancer analysis

Run commands from `breast_cancer_analysis` using its tuned dataset config.

## XAI node and prediction-group selection

Run these commands from `breast_cancer_analysis`:

```bash
# Default: the highest-probability likely-positive prediction (LP).
python run_pipeline.py --stage xai --folder data_corr_0.33_knn_n_3000_n_neg_10000

# Explicit graph node ID overrides automatic prediction selection.
python run_pipeline.py --stage xai --folder data_corr_0.33_knn_n_3000_n_neg_10000 --node-id 7980 --xai-group LP

# Highest-probability predicted negative (RN).
python run_pipeline.py --stage xai --folder data_corr_0.33_knn_n_3000_n_neg_10000 --xai-group RN

# Explain the top five predictions in the chosen group.
python run_pipeline.py --stage xai --folder data_corr_0.33_knn_n_3000_n_neg_10000 --xai-group LP --top-k 5
python run_pipeline.py --stage xai --folder data_corr_0.33_knn_n_3000_n_neg_10000 --xai-group RN --top-k 5
```

Node IDs are dataset-specific; replace `7980` with the intended graph ID for
that dataset. `--xai-group` also accepts the alias `--prediction-group`. `LP`
reads `generated/data/pred_likely_pos_bc_PNA.csv`; `RN` reads
`generated/data/pred_negatives_PNA.csv` (respecting model-specific output paths).
Without an explicit ID, rows are ranked by descending `pred_probs`; ties keep
file order. `--top-k` defaults to 1. Missing exports are reconstructed from the
saved predictions; empty groups or insufficient rows fail with a clear error.
Each selected node gets its own XAI, correlation, Δβ, and three-panel composite
outputs. The current pipeline XAI model is PNA.

The direct runner supports the same flags and additionally `--node-name`:

```bash
python train_gnn/run_xai_explainer.py --config data_corr_0.33_knn_n_3000_n_neg_10000/config.yaml --model PNA --xai-group RN --top-k 3
```

The `--stage accumulate` stage also plots mean and median preprocessed probe degree
for each soft label (0–4), independently of archived training runs. Outputs are
`accumulated_results/plots/degree_per_soft_label.{pdf,svg,png}` (PNG at 600 dpi)
and `accumulated_results/data/degree_per_soft_label.csv` (count, mean, median).
To generate only this plot for one dataset folder, run from this analysis directory:
```bash
python evaluation/plot_degree_per_soft_label.py --folder /path/to/data_folder
```

BC uses the same vector composite layouts as AML. Install plotting dependencies
in the active project environment before running either stage:
```bash
python -m pip install -r train_gnn/requirements-plotting.txt
python run_pipeline.py --stage xai --folder data_corr_0.33_knn_n_3000_n_neg_10000 --node-id 8858 --xai-group LP
python run_pipeline.py --stage accumulate --folder data_corr_0.33_knn_n_3000_n_neg_10000 --models PNA
```
XAI saves `generated/plots/XAI_combined_node_<id>_PNA.pdf` and `.png`:
LP combines the explanation subgraph, neighbor correlation, and current/external
Δβ; RN combines the explanation subgraph and all-seed correlation.
Accumulate saves `accumulated_results/plots/UMAP_before_PNA_5_runs.{pdf,png}`
and `UMAP_after_PNA_5_runs.{pdf,png}`, using the five archived run PDFs.
