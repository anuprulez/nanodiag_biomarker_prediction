# AML analysis

Run commands from `aml_analysis` in the project environment. New datasets require
preprocessing and tuning before training; downstream stages use the dataset's
tuned `config.yaml` and AML input matrix/sample metadata.

```bash
python run_pipeline.py --stage preprocess
python run_pipeline.py --stage tune
python run_pipeline.py --stage train --folder data_corr_0.33_knn_n_3000_n_neg_10000
```

Each successful pipeline training invocation saves the next archive under
`<dataset>/runs/run1/`, `run2/`, and so on, with `config.yaml`, a `run.json` model
manifest, `generated/` artifacts, and `logs/train.log`. Current files remain in
`<dataset>/generated/`. Failed training does not publish a completed archive.
Direct calls to `train_gnn/main.py` do not create run archives.

## Accumulation

```bash
python run_pipeline.py --stage accumulate --folder data_corr_0.33_knn_n_3000_n_neg_10000
python run_pipeline.py --stage accumulate --models PNA XGBoost
```

Accumulation reads completed archives without training and saves outputs in
`<dataset>/accumulated_results/`, alongside `evaluation/`:

- `plots/`: average metric heatmap, radar plot, mean/std learning curves, and
  PNA likely-positive CpG overlap pie chart in PNG/PDF.
- `data/`: metric/history summaries with contributing run counts, included-run
  provenance, CpG membership/count CSVs, and overlap JSON.

The overlap chart reads `pred_likely_pos_aml_PNA.csv` across archived runs. It
counts unique CpGs appearing in exactly one, two, or more included runs and
reports the all-run intersection. Multiple gene annotations count once per CpG
per run. Missing exports are reported and excluded. Different early-stopping
lengths use the available runs at each epoch. No completed archives means
cumulative plotting is skipped.

## Compare current or archived predictions

```bash
# Default: current generated/ artifacts.
python run_pipeline.py --stage compare-predictions \
  --folder data_corr_0.33_knn_n_3000_n_neg_10000 --models PNA

# Saved run artifacts and config; --run 2 is equivalent to --run run2.
python run_pipeline.py --stage compare-predictions \
  --folder data_corr_0.33_knn_n_3000_n_neg_10000 --run run2 --models PNA
```

Current comparisons write to `<dataset>/evaluation/`; archived comparisons write
to `<dataset>/evaluation/run2/`. Without `--models`, comparison selects all tuned
models, or the models recorded in the selected run. A missing run/model raises
an error rather than falling back to current artifacts. `--run` is also supported
by `evaluation/run_comparisons.py`.

Predicted-negative external plots and derived tables use **predicted class 5
only**, including older archives exporting classes 3–5. Source files are
preserved. Likely-positive selections are unchanged. External AML statistics
and cohort inputs remain AML-specific; see [evaluation details](evaluation/README.md).

Comparison also generates cumulative outputs in `accumulated_results/`. The
`all` stage reaches accumulation through comparison without invoking it twice.
Omit `--folder` to process configured dataset folders; add `--dry-run` to preview
commands without changing files. With no folder, a requested run must exist in
each selected dataset. The AML-specific `--stage compare_data_overlap` remains
available.

Regenerate plots to apply the updated styling: generic GNN embedding titles,
36 pt main external-comparison title, consistent 16 pt axis labels, and enlarged
numeric annotations and ticks.

Accumulation also creates one box plot for each accuracy heatmap metric:
validation/test accuracy, macro/weighted/micro F1, weighted precision, and
weighted recall. Each figure compares the selected models using one observation
per available run, with individual points and contributing counts. Figures are
saved as `accumulated_results/plots/boxplot_<metric>_across_runs.png/.pdf` and the
underlying values as `accumulated_results/data/accuracy_metrics_by_run.csv`.
Missing/non-finite values are excluded; models with one run show a single point.

## XAI node and prediction-group selection

Run these commands from `aml_analysis`:

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
reads `generated/data/pred_likely_pos_aml_PNA.csv`; `RN` reads
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
