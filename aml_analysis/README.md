# AML analysis

Pipeline for methylation-based CpG prediction, explanation and cohort follow-up.
Run commands from `aml_analysis/` in the project environment. Relative dataset
folders are resolved under this directory; absolute paths are also accepted.

## Commands

Run stages in this order:

`preprocess → tune → train → xai → compare-predictions → accumulate → post-evaluation`

The commands below use one dataset and PNA, the model required for the fixed-panel
post-evaluation. Add other models to `--models` when needed.

```bash
dataset_folder=data_corr_0.33_knn_n_3000_n_neg_10000
```

### 1. Preprocess

```bash
python run_pipeline.py --stage preprocess --folder "$dataset_folder"
```

Reads preprocessing settings from `config/config.yaml`, prepares the graph and
fixed train/validation/test splits, and validates the outputs. Original methylation,
annotation, seed and sample-metadata inputs must be available at the configured
paths. The saved dataset preprocessing settings are in `preprocessing_config.yaml`.

### 2. Tune

```bash
python run_pipeline.py --stage tune --folder "$dataset_folder" --models PNA --trials 25
```

Tuning uses training/validation partitions and writes the dataset's tuned
`config.yaml`. `--trials` adds trials per selected model per invocation; repeated
calls resume the studies. Use `--timeout` for a time limit and `--search-space`
for another search configuration. See [hyperparameter optimization](HYPERPARAMETER_OPTIMIZATION.md).

### 3. Train and archive runs

```bash
python run_pipeline.py --stage train --folder "$dataset_folder" --models PNA
```

Each successful invocation creates the next archive under `runs/run1/`,
`runs/run2/`, and so on, retaining the run configuration, model manifest, generated
artifacts and training log. Current outputs remain in `generated/`.

The fixed 17-CpG follow-up requires five completed PNA runs. For a new dataset,
run the training command **five times in total**, then accumulate those runs.
Check existing archives before adding runs; the command always creates another
archive. Direct calls to `train_gnn/main.py` do not create these archives.

### 4. Explain predictions

Prepare the external methylation reference once before external comparisons or
explanation figures that use it. These helpers reuse downloaded files:

```bash
python evaluation/download_datasets.py
python evaluation/evaluate_datasets.py
python evaluation/validate_results.py
```

Then explain the top likely-positive prediction:

```bash
python run_pipeline.py --stage xai --folder "$dataset_folder"
```

Useful alternatives:

```bash
python run_pipeline.py --stage xai --folder "$dataset_folder" --xai-group LP --top-k 5
python run_pipeline.py --stage xai --folder "$dataset_folder" --xai-group RN --top-k 5
```

`LP` selects likely-positive predictions; `RN` selects predicted negatives.
`--prediction-group` is an alias for `--xai-group`. Ranking uses descending
`pred_probs`, preserving file order for ties. `--node-id` overrides automatic
selection; choose a graph ID from the selected dataset. Missing prediction exports
are reconstructed from saved predictions; empty or insufficient groups fail
explicitly. XAI currently uses PNA.

Per-node composites are written to
`generated/plots/XAI_combined_node_<id>_PNA.{pdf,png}`. LP combines the explanation
subgraph, neighbor correlation and original/external delta beta; RN combines the
explanation subgraph and all-seed correlation.

### 5. Compare predictions

```bash
python run_pipeline.py --stage compare-predictions --folder "$dataset_folder" --models PNA
```

This compares current predictions with the external methylation reference and
updates cumulative summaries. To compare an archived run instead:

```bash
python run_pipeline.py --stage compare-predictions \
  --folder "$dataset_folder" --models PNA --run run1
```

Current results go to `evaluation/`; archived results go to `evaluation/run1/`.
`--run` is supported only for prediction comparison. Missing runs or models
raise an error. Predicted-negative external comparisons use predicted class 5.

### 6. Accumulate completed runs

```bash
python run_pipeline.py --stage accumulate --folder "$dataset_folder" --models PNA
```

Reads completed archives, rebuilds missing archived evaluation tables, and writes
`accumulated_results/`. It does not train models. Outputs include run-level
metrics and provenance, learning curves, overlap summaries, accuracy box plots,
mean/median degree by soft label, and composite UMAP figures. CpGs with multiple
gene annotations count once per run in overlap summaries.

Prediction comparison also updates cumulative outputs. Run `accumulate` explicitly
to regenerate all archived comparisons and summaries before post-evaluation.

### 7. Post-evaluation

```bash
python run_pipeline.py --stage post-evaluation --folder "$dataset_folder" --models PNA
```

Requires the completed five-run PNA evaluation and the exact frozen **17-CpG
panel**. Candidates are common to five runs and selected by external
|delta beta| ≥ 0.30, P < 0.01, FDR < 0.01 and matching nonzero internal/external
direction. A different panel size or identity fails rather than being substituted.

The stage verifies original/external methylation effects, produces genomic
annotation and Reactome enrichment when resources are available, renders the
current candidate-only patient comparison, and runs the complete cross-cohort
methylation/expression workflow.

The active patient comparison displays **Random Forest, XGBoost, TabPFN and
TabICL**, using original-patient five-fold CV and transfer from models fitted on
all original patients to the external cohort. Its combined PDF contains ROC,
precision–recall and confusion-matrix pages. No external fitting or tuning is
performed for this patient comparison.

To run only the download-to-plot methylation/expression analysis:

```bash
python run_pipeline.py --stage post-evaluation \
  --folder "$dataset_folder" --post-evaluation-analysis cross-cohort
```

This mode uses the frozen panel and original methylation inputs, without running
the other patient-classifier analyses. It downloads missing expression resources,
prepares matrices, fits limma, maps genes to CpGs, calculates direction concordance
and correlations, validates results, and exports PDF, editable SVG and 300-dpi PNG.
No manual download or R command is required between these steps.

### Methylation/expression publication figures

The post-evaluation stage also creates candidate-panel adaptations of Figures 2,
3, 4B/C and 6 from [Liu et al., 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7263199/)
under `<dataset>/accumulated_results/post_evaluation/methylation_expression/publication_figures/`.
The existing cross-cohort scatterplot omits correlation coefficients and their
P values and uses the requested two-line title with Δβ.

The additional plots show genomic context, disease-effect heatmaps, original and
external methylation heatmaps, and candidate-locus annotations. These are adapted
to the saved data: Figure 3 is **not** matched-patient cis/trans correlation,
Figure 4C shows the external cohort **rather than other cancer types**, and
Figure 6 shows annotations/effects **without ATAC-seq or ChIP-seq signal**.
See the generated `publication_figures/README.md` for methods and limitations.
AML retains all 17 CpGs in methylation/annotation plots and marks unavailable
expression explicitly (16/17 evaluable). Outputs include PDF, editable SVG,
300-dpi PNG, source tables, clustering orders and a manifest.

Run these with the standard stage, or use `--post-evaluation-analysis cross-cohort`
to avoid rerunning patient classifiers. Existing source downloads are reused.

### NIAPU rankings comparison and UMAP

```bash
python run_pipeline.py --stage niapu-rankings --folder "$dataset_folder" --run run1
```

The patient-classifier comparison needs preprocessing outputs and the external
methylation reference prepared above. For its feature selection, it reads the
dataset's `p_out_gene_rankings` directly and selects **NIAPU class 2
(likely-positive) nodes only**, without GNN predictions, top-N probability
thresholds or a five-training-run intersection. This classifier portion does not
require graph training or the fixed 33/17-CpG post-evaluation. The four UMAP
panels additionally require an archived PNA training run (default `run1`); use
`--skip-umap` for classifiers alone. A tuned `config.yaml` is used when available;
otherwise the saved `preprocessing_config.yaml` is used.

#### Four matched UMAP panels

```bash
# Plot only, using the archived best-checkpoint PNA predictions from run1.
# naipu-rankings is accepted as an alias for niapu-rankings.
python run_pipeline.py --stage naipu-rankings --folder "$dataset_folder" --run run1 --umap-only

# Existing likely-positive patient-classifier comparison without UMAP/PNA inputs.
python run_pipeline.py --stage niapu-rankings --folder "$dataset_folder" --skip-umap
```

The default full stage creates UMAPs followed by the patient-classifier comparison.
`--umap-only` needs preprocessing rankings plus one completed PNA run, with no
external download or patient-classifier refitting. Choose another archived run
with `--run run2` (or `--run 2`). The saved PNA predictions were generated after
restoring that run's best validation checkpoint. The script checks them against
raw saved prediction, true-label and node-ID tensors and records checkpoint/input
hashes. It neither trains another PNA model nor uses training-node predictions.

The four panels use the **same held-out CpG–gene nodes and all five classes**:

1. NIAPU labels, NeDBIT features.
2. NIAPU labels, NeDBIT + DNA methylation features.
3. NIAPU labels, DNA methylation features.
4. PNA predicted labels, final PNA message-passing layer representation
   (256 dimensions in run1), projected with UMAP.

Colors follow the existing training UMAPs (`tab10`): positive seed blue, likely
positive orange, weakly negative green, likely negative red, reliable negative
purple, with opacity 0.9.

Panel 4 uses a separate UMAP of the saved final PNA message-passing output `h`
(after the last residual PNAConv and LayerNorm), from
`runs/<run>/generated/models/embed_conv_<disease>_PNA.pt`. The model was trained
with NeDBIT + DNA methylation. Saved latent values enter UMAP without additional
scaling, using the same UMAP settings as panels 1–3. Its row order matches the
verified held-out node IDs. This is the final GNN representation, rather than
class probabilities or the concatenated raw features and intermediate states. NeDBIT
columns come from the archived run's `keep_feature_names`; methylation columns are
patient measurements describing CpG nodes. Labels are excluded from all feature
matrices and never passed to UMAP. Columns are median-imputed when necessary and
standardized on plotted nodes; each block is divided by the square root of its
nonconstant feature count to balance the two blocks. UMAP settings are read from the selected archived run's configuration, matching
the earlier training UMAPs. For `run1` in both datasets: correlation distance,
20 neighbors, `min_dist=0.8`, seed 32. Trustworthiness uses this input metric;
silhouette and Davies–Bouldin scores use Euclidean distances.

Figures omit scores and subtitles. All held-out nodes are included, without a
likely-positive filter, and panels 1–3 use NIAPU labels while panel 4 uses PNA predicted labels. The quality
TSV retains **silhouette in 2D ↑, silhouette in feature/latent space ↑,
Davies–Bouldin in 2D ↓, and trustworthiness (15 neighbors) ↑**. The quality table
also includes Calinski–Harabasz in 2D ↑ and NIAPU/PNA adjusted Rand agreement.
Feature-space silhouette checks whether apparent separation survives outside the
projection. These are descriptive scores of supplied labels, not disease
accuracy or independent biological validation: NIAPU and NeDBIT share graph
information, and PNA is trained to predict NIAPU. CpG–gene nodes can share a CpG.

UMAP outputs are under `accumulated_results/niapu_comparison/`:

- `plots/niapu_umap_comparison.{pdf,svg,png}`: four-panel figure; individual
  `niapu_umap_1` through `niapu_umap_4` plots are exported too.
- `data/niapu_umap_quality.tsv` and `data/niapu_umap_coordinates.tsv`.
- `umap_manifest.json`: selected run, feature columns, settings and input hashes.
- `UMAP_REPORT.md`: methods and interpretation; `cache/niapu_umap_run*.npz`:
  coordinates reused when input hashes/settings match; final-layer coordinates
  have a separate `niapu_umap_pna_last_layer_run*.npz` cache. Selecting a different run
  replaces displayed figures/tables while retaining per-run coordinate caches.

The all-class UMAP comparison is separate from the **class-2-only** external
patient-classifier feature selection described below.

The merged ranking file combines the disjoint preprocessing splits. Multiple
class-2 gene annotations are deduplicated to one CpG; its maximum NIAPU score is
retained for ranking/display, with no score cutoff. "Common" means NIAPU likely
positives measured in the external reference. Sites must meet **all** criteria:

- |external delta beta| ≥ 0.30;
- external Welch P < 0.01;
- BH FDR < 0.01, computed over the complete shared external CpG universe before
  selecting NIAPU candidates;
- matching nonzero original/external methylation directions.

Original delta beta is recomputed from original patient methylation. The selected
panel size is determined by these criteria; it is not forced to 33 or 17.
The same patient classifiers—**Random Forest, XGBoost, TabPFN and TabICL**—use
the existing fixed hyperparameters/checkpoints, original-patient five-fold CV,
and final fitting on all original patients before external prediction. Targets
are disease vs healthy, not NIAPU classes. No class weighting or external fitting
is applied. Omit `--models`: this stage has a fixed four-classifier suite.

Results are isolated in `<dataset>/accumulated_results/niapu_comparison/`:

```text
niapu_comparison/
  REPORT.md
  CLASSIFIER_COMPARISON.md
  manifest.json
  selection_manifest.json
  classifier_manifest.json
  validation.json
  candidate_cohort_comparison.pdf
  data/
    niapu_likely_positive_nodes.tsv
    niapu_likely_positive_unique_CpGs.tsv
    all_common_likely_positive_CpGs.tsv
    selected_niapu_CpGs.tsv
    external_full_test_universe.tsv.gz
    candidate_cohort_performance.tsv
    candidate_cohort_predictions.tsv
  plots/candidate_cohort_*.{pdf,svg,png}
  models/
  cache/
```

Accuracy at threshold 0.5, balanced accuracy, sensitivity, specificity, F1, AUROC,
AUPRC, Brier, confusion counts and confidence intervals are reported separately
for original CV and external transfer. The report records selection counts and
source hashes. Matrix and prediction caches are reused when their fingerprints
match. Logs are at `<dataset>/logs/niapu-rankings.log`. Existing post-evaluation
results are not changed. Without `--folder`, datasets with NIAPU rankings and a
saved dataset/preprocessing config are selected.

Because external labels are used to select CpGs, external accuracy is
**exploratory and selection-biased**, not independent validation. Original-patient
CV is also conditional on methylation-based NIAPU/feature selection performed
before CV. Intervals do not correct for that selection or study confounding.

### Other supported commands

Preview a command without running stages or modifying outputs:

```bash
python run_pipeline.py --stage post-evaluation --folder "$dataset_folder" --dry-run
```

Compare **run1** validation/test scores across preprocessing folders:

```bash
python run_pipeline.py --stage compare-preprocess --folders \
  data_corr_0.33_knn_n_3000_n_neg_10000 \
  data_corr_0.2_knn_n_3000_n_neg_10000
```

Outputs go to `evaluation/preprocessing_comparison/`; missing run1 scores appear
as N/A. Omit `--folders` to compare matching folders.

`--stage all` runs preprocess, tune, train, XAI and prediction comparison once,
in that order. It does **not** create five training runs or run the explicit
`accumulate`, `post-evaluation` and `niapu-rankings` stages. Omitting `--folder` can process multiple
configured datasets and, for upstream stages, restart outputs; use an explicit
folder for reproducible work on one dataset.

AML also supports preprocessing-input overlap comparison:

```bash
python run_pipeline.py --stage compare_data_overlap --folder "$dataset_folder"
```

## Data sources and cross-cohort methods

| Role | Dataset | Samples used |
| --- | --- | --- |
| Original methylation cases | [GSE58477](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE58477) | 62 AML samples |
| Original methylation controls | [GSE50660](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE50660) | 464 blood controls |
| External methylation controls | [GSE40279](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE40279) | 656 controls |
| External methylation cases | [GSE62298](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE62298) | 68 AML samples |
| Independent expression | [GSE15061](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE15061) | 202 AML, 69 non-leukemic marrow samples |

The plot uses **X = GSE15061 expression log2FC** and **Y = original methylation
delta beta: GSE58477 AML62 minus GSE50660 control464**. External methylation
contrasts are sensitivity results only.

GSE15061 has two processed representations of each biological specimen. Only
**DS** records are analyzed; duplicate DQN3 records and MDS specimens are excluded.
DS values are log2 transformed, quantile normalized and averaged across probes
with unambiguous gene mappings. Limma fits **group + study center** for primary
results and a group-only sensitivity model, with genome-wide BH FDR. Controls
are non-leukemic marrow specimens, not necessarily healthy volunteers.

GPL570 and an HGNC snapshot are downloaded independently of BC. Platform
extraction stops after the complete annotation table. **cg07911080–PRHOXNB/URAD**
lacks an unambiguous expression probe; it remains missing, so **16 of the frozen
17 CpGs** are plotted. FAM65B maps to RIPOR2.

The figure uses centered X=0, promoter/gene-body symbols, enlarged fonts and
arrows labeling only inverse-sign candidates. Its stem is
`AML_17_CpGs_methylation_vs_GSE15061_expression` under
`methylation_expression/cross_cohort_comparison/plots/`.

## Dataset comparability audit

See the [dataset comparability report](evaluation/dataset_comparability/REPORT.md)
for the verified study/specimen differences, complete study–disease confounding,
and a sensitivity analysis using GSE58477's 10 currently excluded CD34+ controls.
All 17 frozen CpGs retain their methylation-effect direction with these controls;
16 pass targeted BH FDR <0.05 across the 17 tests. This reuses the same AML cases
and is a post-selection sensitivity analysis, not independent validation.
The report and audit outputs do not replace existing models or figures.

## Setup and configuration

Use the project ML environment, including compatible PyTorch/PyG and the selected
model packages. Requirements are split by purpose:

- [Benchmark/model dependencies](train_gnn/requirements-benchmark.txt).
- [PyG sampling dependencies](train_gnn/requirements-sampling.txt).
- [Optuna](train_gnn/requirements-optuna.txt).
- [Plot composition](train_gnn/requirements-plotting.txt).

Install additions in the existing environment as needed:

```bash
python -m pip install -r train_gnn/requirements-optuna.txt
python -m pip install -r train_gnn/requirements-plotting.txt
```

`config/config.yaml` controls preprocessing. `config/optuna.yaml` supplies search
spaces and runtime defaults. Downstream stages use `<dataset>/config.yaml` created
by tuning. Foundation-model checkpoints must be available or accessible to the
configured checkpoint downloader; their hashes and settings are recorded.

Cross-cohort analysis also requires numpy, pandas, scipy ≥ 1.11, matplotlib,
scikit-learn, requests, OmegaConf, and **R with limma and statmod**. A dedicated
environment for only that analysis is available:

```bash
conda env create -f ../post_evaluation_support/environment-cross-cohort.yml
conda activate nanodiag-cross-cohort
```

This dedicated environment does not replace the full GNN training environment.
Rscript discovery checks `--rscript`, `RSCRIPT`, `PATH`, then an existing ancestor's
`.analysis_envs/bc-limma/bin/Rscript`. Use an absolute `--rscript` path for another
installation. Missing dependencies fail explicitly.

## Downloads, caches and reruns

| Post-evaluation option | Behavior |
| --- | --- |
| `--post-evaluation-analysis all` | Default: full post-evaluation, including cross-cohort analysis. |
| `--post-evaluation-analysis cross-cohort` | Run only cross-cohort preparation, modeling, validation and plotting. |
| `--offline` | Use cached resources only; missing or incompatible caches fail. |
| `--refresh-expression` | Rebuild expression matrices and refit limma, reusing downloaded sources. |
| `--rscript /path/to/Rscript` | Select the R executable. |
| `--pathway-gmt /path/to/ReactomePathways.gmt` | Use a local pathway resource. |
| `--random-panels N` | Number of matched random panels; default 30, minimum 2. |
| `--background-size N` | Sampled CpG background; default 800, minimum 100. |

Available files are reused **at their expected cache paths**. Missing resources
are downloaded automatically; BC query caches must match the requested sample
and gene IDs. A final plot or processed matrix alone does not replace required
source caches. Interrupted downloads cannot become completed cache files.

Cross-cohort preprocessing and limma results are reused only when their input/output hashes
match. Plots and numerical validation are regenerated. After upgrading R/limma
or changing analysis environments, use `--refresh-expression` to refit.

## Output layout

Paths below are relative to the selected dataset folder:

```text
preprocessing_config.yaml
config.yaml
preprocessed/
generated/
  data/
  models/
  plots/
runs/run<number>/
  config.yaml
  run.json
  generated/
  logs/train.log
evaluation/
accumulated_results/
  data/
  evaluation/
  plots/
  post_evaluation/
    REPORT.md
    manifest.json
    candidate_cohort_comparison.pdf
    candidate_cohort_comparison.json
    data/candidate_cohort_predictions.tsv
    data/candidate_cohort_performance.tsv
    plots/candidate_cohort_*.{pdf,svg,png}
    methylation_expression/
      pipeline_manifest.json
      cross_cohort_comparison/
        REPORT.md
        manifest.json
        validation.json
        data/
        plots/
logs/<stage>.log
```

The cross-cohort `data/` directory contains per-CpG and per-gene results,
coefficient intervals, expression significance and sensitivity statistics.
`pipeline_manifest.json` records completion/failure and cache status; manifests
and model reports record source provenance. AML downloads, normalized matrices,
probe mappings and limma tables are under
`methylation_expression/cross_cohort_comparison/{source,data}/`.

Numerical checks cover frozen CpG identities, original sample-level methylation
contrasts, gene-level BH FDR, coefficient intervals and inverse-only labels.
AML checks also verify unique DS biological samples, missing probe coverage,
and independent numpy OLS agreement with the center-adjusted model.

## Interpretation and further documentation

Candidates were selected using external methylation labels, so subsequent
analyses are exploratory rather than independent validation. Patient CV does not
remove prior feature-selection bias. Case/control tissue and study differences
can influence effects. Cross-cohort delta beta versus log2FC compares features;
it is not a correlation between methylation and expression across matched
patients and does not establish causation. The original AML methylation contrast completely confounds disease with
study and cell composition. Expression center adjustment does not repair that
confounding; adjusted and unadjusted expression directions can differ.

See [external methylation evaluation](evaluation/README.md) for reference-cohort
preparation and [shared post-evaluation methods](../post_evaluation_support/README.md)
for detailed modeling and statistical procedures. Generated `REPORT.md` and
validation manifests describe the actual run.
