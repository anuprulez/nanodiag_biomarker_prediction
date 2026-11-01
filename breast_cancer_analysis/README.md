# Breast cancer analysis

Pipeline for methylation-based CpG prediction, explanation and cohort follow-up.
Run commands from `breast_cancer_analysis/` in the project environment. Relative dataset
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

The fixed 33-CpG follow-up requires five completed PNA runs. For a new dataset,
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

Requires the completed five-run PNA evaluation and the exact frozen **33-CpG
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
`accumulate` and `post-evaluation` stages. Omitting `--folder` can process multiple
configured datasets and, for upstream stages, restart outputs; use an explicit
folder for reproducible work on one dataset.

## Data sources and cross-cohort methods

| Role | Dataset | Samples used |
| --- | --- | --- |
| Original methylation | [GSE66695](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE66695) | 80 tumors, 40 normals |
| External methylation controls | [GSE101961](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE101961) | 121 controls |
| External methylation cases | [GSE226569](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE226569) | 694 tumors |
| Independent expression | [TCGA-BRCA HiSeqV2, UCSC Xena](https://xenabrowser.net/datapages/?dataset=TCGA.BRCA.sampleMap%2FHiSeqV2&host=https%3A%2F%2Ftcga.xenahubs.net) | Selected set: 775 tumors, 84 normals |

The 33-point plot uses **X = TCGA-BRCA expression log2FC** and
**Y = original GSE66695 tumor-minus-normal methylation delta beta**. The supplied
expression scale is log2(RSEM normalized count + 1). Exact sample-type matching
to GDC methylation IDs reconstructs the expression cohort; GDC methylation does
not replace the original delta beta.

Genome-wide expression is fetched in resumable chunks. Limma fits the full
patient-blocked model for the primary X values and a paired-patient sensitivity
model using 77 pairs, with genome-wide variance moderation and BH FDR. Multiple
available annotated-gene coefficients are averaged per CpG; coverage and
mixed-direction flags are retained. C7orf20 maps to GET4.

The figure uses centered X=0, promoter/gene-body symbols, enlarged fonts and
arrows labeling only inverse-sign candidates. Its stem is
`GSE66695_delta_beta_vs_TCGA_expression_log2FC` under
`methylation_expression/cross_cohort_comparison/plots/`.

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
and model reports record source provenance. BC matched query caches are under
`methylation_expression/source/`; genome-wide chunks and limma outputs are under
`methylation_expression/direction_concordance/`.

Numerical checks cover frozen CpG identities, original sample-level methylation
contrasts, gene-level BH FDR, coefficient intervals and inverse-only labels.
BC patient-blocked coefficients are checked independently with numpy GLS.

## Interpretation and further documentation

Candidates were selected using external methylation labels, so subsequent
analyses are exploratory rather than independent validation. Patient CV does not
remove prior feature-selection bias. Case/control tissue and study differences
can influence effects. Cross-cohort delta beta versus log2FC compares features;
it is not a correlation between methylation and expression across matched
patients and does not establish causation. Annotation-based mappings are candidate regulatory hypotheses, not confirmed targets.

See [external methylation evaluation](evaluation/README.md) for reference-cohort
preparation and [shared post-evaluation methods](../post_evaluation_support/README.md)
for detailed modeling and statistical procedures. Generated `REPORT.md` and
validation manifests describe the actual run.
