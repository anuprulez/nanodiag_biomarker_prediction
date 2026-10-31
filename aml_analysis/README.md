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

The `--stage accumulate` stage also plots mean and median preprocessed probe degree
for each soft label (0–4), independently of archived training runs. Outputs are
`accumulated_results/plots/degree_per_soft_label.{pdf,svg,png}` (PNG at 600 dpi)
and `accumulated_results/data/degree_per_soft_label.csv` (count, mean, median).
To generate only this plot for one dataset folder, run from this analysis directory:
```bash
python evaluation/plot_degree_per_soft_label.py --folder /path/to/data_folder
```

Compare run1 validation/test accuracy across preprocessing folders with:

```bash
python run_pipeline.py --stage compare-preprocess --folders \
  data_corr_0.33_knn_n_3000_n_neg_10000 \
  data_corr_0.2_knn_n_3000_n_neg_10000 \
  data_corr_0.5_knn_n_3000_n_neg_10000 \
  data_corr_0.7_knn_n_3000_n_neg_10000
```

Only run1 heatmap CSVs are used. Missing run1 scores appear as N/A. PDF/SVG/PNG
heatmaps, score TSVs, and source provenance go to `evaluation/preprocessing_comparison/`.
Omit `--folders` to compare all matching folders; use `--models` to select models.
`--stage accumulate` remains separate from this preprocessing comparison.

## Post-evaluation: fixed 17-CpG panel

```bash
python run_pipeline.py --stage post-evaluation --folder data_corr_0.33_knn_n_3000_n_neg_10000 --models PNA
```

This opt-in stage follows up the **17 common likely-positive CpGs** selected from
all five PNA runs using **|external Δβ| ≥ 0.30, external P < 0.01, external FDR < 0.01,
and matching nonzero direction**. It verifies the exact panel size and recomputes
patient-level original/external effects from raw beta values.

Outputs in `<dataset>/accumulated_results/post_evaluation/` include:

- `REPORT.md` and `manifest.json`: detailed results, methods, input hashes and limitations.
- PNG/PDF effect-size forest plots, patient heatmaps, per-CpG distributions and PCA.
- Patient-level seed-only, candidate-only and combined-panel ROC and PR;
  AUROC/AUPRC/Brier/sensitivity, conditional bootstrap intervals and paired incremental AUROC.
  The external performance tables additionally compare all **259 common likely-positive**
  and **218 common predicted-negative** CpGs from the five-run Δβ tables,
  without effect-size/significance filtering. Both added models classify healthy
  versus disease patients using original-cohort training and external evaluation.
- Training-fold-ranked panel-size curves and 30 mean/SD-matched random-panel comparisons.
- Chromosome, gene-region and CpG-island annotation plus exploratory CpG-level
  Reactome pathway enrichment; explicit status for any unavailable resources.
- Auditable candidate lists, patient matrices/annotations, patient folds, predictions,
  matching memberships, panel coverage and pathway tables in `data/`.

This stage is **not included in `--stage all`** because it requires the completed
five-run evaluation. It supports `--dry-run`, `--random-panels`, `--background-size`,
`--pathway-gmt` and `--offline`. Without `--folder`, only folders with accumulated
PNA evaluation tables are considered. It does not retrain graph models or change seeds.

**Interpretation:** these candidates were selected using the external cohort;
these analyses are exploratory, not independent validation. External case/control
status is confounded with study (also true of the original AML cohorts). Patient CV
cannot undo prior panel selection. Missing clinical outcomes are documented rather than inferred. Gene-region and
CpG-island context is extracted from the full local GPL13534 platform source;
missing resources are reported explicitly.

See [the shared post-evaluation methods and artifact guide](../post_evaluation_support/README.md)
for precise modeling, bootstrap, random matching, pathway background and validation details.

External predicted-negative CpG distributions are also saved as a single
`plots/external_predicted_negative_cpg_distributions.pdf` (plus SVG/PNG), using
the same complete five-run negative panel as the external classifier: 70 BC
or 218 AML CpGs. Blue/orange points and boxes show healthy/disease patients,
with no footer. The stage also exports the exact CpG list and raw beta table.

The external ROC/PR figure displays only **Seeds only** and **Candidates only**.
Combined and full common LP/negative classifier results remain in the data tables.

### Logistic regression, Random Forest and XGBoost

Post-evaluation compares these three patient classifiers using identical seed-only
and candidate-only features, original-patient training/CV and external transfer.
The external performance figure shows seed ROC/PR and candidate ROC/PR, each with
three model curves. No external fitting or tuning is performed. See
`accumulated_results/post_evaluation/CLASSIFIER_COMPARISON.md` for results,
`data/classifier_*.tsv` for predictions/metrics/paired comparisons, and
`plots/external_classifier_confusion_matrices.pdf` for predictions at threshold 0.50.
The fitted pipelines are saved under `models/` with exact feature order in
`classifier_comparison_manifest.json`.

To run just this comparison from existing validated caches:

```bash
python evaluation/compare_patient_classifiers.py --config data_corr_0.33_knn_n_3000_n_neg_10000/config.yaml
```

Full methods and fixed hyperparameters are in the shared post-evaluation README.
These analyses remain exploratory because of panel selection and study confounding.

### Inverse class weighting

```bash
python evaluation/compare_patient_classifiers.py --config data_corr_0.33_knn_n_3000_n_neg_10000/config.yaml --inverse-class-weighting
```

Uses original-training weights `N_train/(2*n_class)` for all three models,
recomputed separately in every original-patient CV fold. External/test labels
never determine weights. Weighted models, results and new separate ROC, PRC
and confusion plots are saved under
`accumulated_results/post_evaluation/inverse_class_weighted/`; unweighted results
remain available. The full post-evaluation stage now creates both comparisons.
The weighted manifest and `data/training_class_weights.tsv` record the exact
weights. Patient classification threshold remains 0.50.

### Matching weighted and unweighted plot sets

Both sets use the same external patients, seed/candidate panels and all three
classifiers, with identical styling and explicit weighting labels:

- `accumulated_results/post_evaluation/unweighted/external_classifier_plot_set.pdf`
- `accumulated_results/post_evaluation/inverse_class_weighted/external_classifier_plot_set.pdf`

Each PDF contains three pages: ROC, PRC and confusion matrices. Individual
PDF/SVG/PNG figures are in the respective `plots/` subdirectory. The sets are
rendered from the corresponding saved predictions, without refitting models.

### TabPFN3 and TabICLv2

Both patient-classifier comparison sets now include five models: logistic
regression, Random Forest, XGBoost, **TabPFN3** and **TabICLv2**. The exact original
analysis checkpoints and frozen model overrides are used, with the same patient
folds and CpG panels. Checkpoint hashes, versions and parameters are recorded.

```bash
python evaluation/compare_foundation_classifiers.py --config data_corr_0.33_knn_n_3000_n_neg_10000/config.yaml
```

Full post-evaluation adds these models automatically. Weighted foundation results
use training-prior probability correction because these model APIs do not support
sample-weighted fitting; this differs from the loss/split weighting of the existing
three classifiers and is documented explicitly in the shared methods README.
Both matched three-page PDF sets, individual ROC/PRC/confusion figures and result
tables are updated. Cached predictions avoid repeating unchanged GPU inference.

### Current candidate-only cohort comparison

The current five-model comparison uses only the fixed candidate panel: 33 CpGs for breast cancer and 17 for AML. Original-cohort five-fold patient cross-validation and external predictions from original-trained models are compared using blue and vermilion, matching the delta-beta forest plot. There are no separate weighted/unweighted figure sets. The active outputs are `accumulated_results/post_evaluation/candidate_cohort_comparison.pdf` and `plots/candidate_cohort_{ROC,PRC,confusion_matrices}.{pdf,svg,png}`. These supersede the older seed-panel and weighting comparisons described above. Historical outputs are moved to `accumulated_results/retired_classifier_comparisons/`.
