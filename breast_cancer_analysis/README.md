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

Compare run1 validation/test accuracy across preprocessing folders with:

```bash
python run_pipeline.py --stage compare-preprocess --folders \
  data_corr_0.33_knn_n_3000_n_neg_10000 \
  data_corr_0.2_knn_n_3000_n_neg_10000 \
  data_corr_0.5_knn_n_3000_n_neg_10000 \
  data_corr_0.7_knn_n_3000_n_neg_10000
```

```bash
python run_pipeline.py --stage compare-preprocess --folders data_corr_0.33_knn_n_3000_n_neg_10000/ data_corr_0.2_knn_n_3000_n_neg_10000/ data_corr_0.5_knn_n_3000_n_neg_10000/ data_corr_0.7_knn_n_3000_n_neg_10000/ data_corr_0.33_knn_n_500_n_neg_10000/ data_corr_0.33_knn_n_1500_n_neg_10000/ data_corr_0.33_knn_n_4500_n_neg_10000/ data_corr_0.33_knn_n_3000_n_neg_5000/ data_corr_0.33_knn_n_500_n_neg_15000/ data_corr_0.33_knn_n_1500_n_neg_15000/
```

Only run1 heatmap CSVs are used. Missing run1 scores appear as N/A. PDF/SVG/PNG
heatmaps, score TSVs, and source provenance go to `evaluation/preprocessing_comparison/`.
Omit `--folders` to compare all matching folders; use `--models` to select models.
`--stage accumulate` remains separate from this preprocessing comparison.

## Post-evaluation: fixed 33-CpG panel

```bash
python run_pipeline.py --stage post-evaluation --folder data_corr_0.33_knn_n_3000_n_neg_10000 --models PNA
```

This opt-in stage follows up the **33 common likely-positive CpGs** selected from
all five PNA runs using **|external Δβ| ≥ 0.30, external P < 0.01, external FDR < 0.01,
and matching nonzero direction**. It verifies the exact panel size and recomputes
patient-level original/external effects from raw beta values.

Outputs in `<dataset>/accumulated_results/post_evaluation/` include:

- `REPORT.md` and `manifest.json`: detailed results, methods, input hashes and limitations.
- PNG/PDF effect-size forest plots, patient heatmaps, per-CpG distributions and PCA.
- Patient-level seed-only, candidate-only and combined-panel ROC and PR;
  AUROC/AUPRC/Brier/sensitivity, conditional bootstrap intervals and paired incremental AUROC.
  The external performance tables additionally compare all **189 common likely-positive**
  and **70 common predicted-negative** CpGs from the five-run Δβ tables,
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
