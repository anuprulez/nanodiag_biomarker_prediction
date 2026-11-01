# GSE40279 versus GSE62298

Healthy: 656 samples. AML: 68 samples.
Shared CpGs: 470,043; CpG–gene rows: 393,997.

Full table: `results/cpg_gene_beta_comparison.tsv`. All shared CpGs, including unannotated probes: `results/shared_cpg_statistics.tsv`.

The preview below contains the first 10 CpG–gene pairs in CpG ID order, not a significance ranking.

| CpG_gene | Healthy GSE40279 (mean ± SD) | AML GSE62298 (mean ± SD) | Δβ (AML − healthy) |
|---|---:|---:|---:|
| cg00000029_RBL2 | 0.476097 ± 0.068449 | 0.552630 ± 0.220065 | +0.076533 |
| cg00000108_C3orf35 | 0.944313 ± 0.020977 | 0.936439 ± 0.012153 | -0.007874 |
| cg00000109_FNDC3B | 0.892913 ± 0.033777 | 0.901500 ± 0.024141 | +0.008587 |
| cg00000236_VDAC3 | 0.746412 ± 0.041170 | 0.737685 ± 0.152562 | -0.008728 |
| cg00000289_ACTN1 | 0.609753 ± 0.065780 | 0.708363 ± 0.081368 | +0.098610 |
| cg00000292_ATP2A1 | 0.826354 ± 0.024053 | 0.769176 ± 0.138833 | -0.057178 |
| cg00000321_SFRP1 | 0.312680 ± 0.067525 | 0.388763 ± 0.165339 | +0.076083 |
| cg00000622_NIPA2 | 0.007836 ± 0.005106 | 0.009879 ± 0.004786 | +0.002043 |
| cg00000658_MAN1B1 | 0.817863 ± 0.023315 | 0.817420 ± 0.028870 | -0.000442 |
| cg00000714_TSEN34 | 0.188988 ± 0.025302 | 0.278197 ± 0.141297 | +0.089209 |

From aml_analysis, run `python evaluation/download_datasets.py` to download/validate the matrices, then `python evaluation/evaluate_datasets.py` to regenerate the results. Run `python evaluation/validate_results.py` to validate output identities, counts, delta values, and independently recompute three CpGs from both source matrices.

Method: exact intersection of cg probe IDs; GPL13534 UCSC_RefGene_Name (legacy hg19) annotation reused from ../raw. Multiple gene assignments are expanded into unique CpG–gene pairs. Published beta values are unchanged. SD uses n−1; missing values are omitted per probe, with valid n reported. No imputation, filtering by effect, or statistical significance threshold was applied.

These are descriptive cross-study differences without batch, age, sex, or cell-composition adjustment. Whole blood controls and primary AML blood differ in cellular composition. GSE62298 is the requested AML GEO dataset; it is not TCGA.

Dataset URLs, checksums, row counts and policies are in results/analysis_report.json; sample metadata and cohort tables are retained in raw/ and results/.

## Dataset-specific prediction comparisons

```bash
python run_pipeline.py --stage compare-predictions --folder data_corr_0.33_knn_n_3000_n_neg_10000
```

Omit `--folder` for all dataset folders, or use `--models PNA GCN` to select
models. Each dataset must have its own tuned `config.yaml`. Comparisons use
`generated/data` test-partition tables and `generated/models` prediction tensors;
flat legacy artifacts remain readable. Existing `generated/data/pred_negatives_<model>.csv` and
`pred_likely_pos_aml_<model>.csv` are authoritative, including any updated XAI
selection rules. Their rows are checked against current saved inference IDs,
feature names, labels and probabilities (allowing float32 CSV rounding). Group
CSVs are copied into evaluation outputs and their original paths/SHA256 hashes
are recorded. Only when both exports are absent are groups reconstructed from
tensors; incomplete or inconsistent exports fail clearly. Editing XAI rules
requires regenerating its group CSVs before comparing. Outputs go to
each dataset's `evaluation/`, including per-model tables, PNG/PDF plots, extracted
groups and checksummed reports. Metadata-aligned original effects are calculated
as AML minus healthy; external cohorts remain GSE40279 and GSE62298.

The standalone `evaluation/run_comparisons.py --config <dataset>/config.yaml`
accepts `--dry-run`. `evaluation/compare_pna_predictions.py` remains a compatible
entry point for the generalized `compare_predictions.py`.

### Enhanced AML prediction comparisons

```bash
python run_pipeline.py --stage compare-predictions --folder data_corr_0.33_knn_n_3000_n_neg_10000
```

All three PNG/PDF figures show **likely positive first, negative second**:
`delta_beta_original_vs_external` (four-panel agreement, absolute effects and
exploratory support), `delta_beta_shift_by_prediction_group`, and
`criterion_sensitivity`. Outputs live in the dataset's `evaluation/` directory.
The negative selection uses original labels 2–5 and predicted label 5; likely
positives use original labels 2–5 and predicted label 2. Exports sort by descending
`pred_probs` with stable ties. Comparison preserves validated custom XAI exports.

Points represent unique CpGs matched to the external comparison table, rather
than all exported CpG–gene rows. Summary TSV/JSON records input rows, unique CpGs,
matched and plotted counts, Pearson/Spearman, direction agreement, MAE, median
absolute effects and fractions above 0.10/0.20. Per-CpG tables include external
Welch p-values, BH FDR, pointwise 95% CI, absolute errors and support flags.

Welch inference uses existing means, sample SDs (ddof=1) and per-CpG valid n.
BH correction uses the complete `evaluation/results/shared_cpg_statistics.tsv`
universe before selecting predictions, including unannotated probes. Untestable
sites remain in the correction family as p=1 but receive missing inference.
The input comparison table is checked for consistency against this universe.
Reference: [SciPy Welch test from summary statistics](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind_from_stats.html).

The exploratory criterion requires absolute external Δβ ≥0.10, BH FDR <0.05,
and consistent nonzero direction. Sensitivity plots/TSVs show effect thresholds
0.05/0.10/0.20 and FDR cutoffs 0.01/0.05/0.10. Denominators are finite matched
unique CpGs; missing significance is not counted as support. Summary effect
threshold fractions use strict >0.10/>0.20. Exact zero-zero sign matches count
as direction agreement, but the support criterion requires nonzero effects.

External controls are GSE40279 and AML samples are GSE62298. Disease and external
study are confounded; age and blood-cell composition may also affect comparisons.
No batch or covariate adjustment is performed. FDR and effect agreement do not
establish independent replication or model superiority over selection-derived
labels. These thresholds are exploratory, not prospectively registered.
Only tuned datasets with completed prediction artifacts can be evaluated; an
explicit `--folder` avoids other folders whose training is still incomplete.

## Archived runs and accumulated results

See the [AML pipeline README](../README.md) for numbered training archives,
`--stage accumulate`, and `--run run2` comparison examples. Accumulation writes
plots and supporting tables under each dataset's `accumulated_results/`, alongside
`evaluation/`. The overlap pie chart uses `pred_likely_pos_aml_PNA.csv` and counts
unique CpGs per run, deduplicating gene annotations.

`compare-predictions` defaults to all tuned models (or the selected run's trained
models). Use `--models PNA` to limit outputs. Without `--run`, it reads current
`generated/` files; with `--run`, it reads that archive and saves comparisons in
`evaluation/run<number>/`. Negative plots and derived comparison tables use
**predicted class 5 only**; archived prediction source files remain unchanged.


## Evaluation accumulation across training runs

Run `python run_pipeline.py --stage accumulate` from the analysis directory
(with the same folder/model selectors as the other stages). This includes
archived comparisons in `<dataset>/evaluation/run1/` through `run5/` and any
additional completed evaluation runs. Generate those comparisons first using
`--stage compare-predictions --run run1` (repeat for each archived run).
Comparisons also refresh accumulated evaluation outputs automatically.

Outputs are in `<dataset>/accumulated_results/evaluation/`, separately for each
model and dataset suffix:

- `summary_*.tsv` and `criterion_sensitivity_*.tsv`: equal-weight means, sample
  standard deviations, and contributing run counts. Sensitivity plots average
  each run's fractions rather than pooling CpGs across runs.
- `delta_beta_criterion_*.png/pdf`: mean support percentage with SD error bars.
- `*_common_unique_cpgs_*.tsv` and `common_cpg_summary_*.tsv`: intersection of
  CpGs across every included run, separately within each prediction group.
  Internal effects are averaged per CpG; external cohort effects and FDR must
  agree across runs and are retained without repeating statistical inference.
- `delta_beta_original_vs_external_*.png/pdf` and
  `delta_beta_shift_by_prediction_group_*.png/pdf`: effects for those common CpGs.
- `included_runs_*.json`: source run names, counts, skipped runs, and methods.

A model/run is included only when its summary, sensitivity table, and both
prediction-group CpG tables exist. Fewer or more than five included runs produce
a warning with the actual count. Models with no complete evaluations are skipped;
an empty CpG intersection skips effect plots while retaining mean results.

Accumulated four-panel figures use consistent 24-point axis titles and tick
labels, 22-point legends, and Pearson-only annotations. PNG exports use at
least 300 DPI; PDF embeds TrueType fonts and SVG retains editable text.

### Five-run PNA UMAP composites

`--stage accumulate` also creates `accumulated_results/plots/UMAP_before_PNA_5_runs`
and `UMAP_after_PNA_5_runs` as vector PDF and at least 300 DPI PNG. Each figure
has five run panels, arranged three above and two centered below. Before uses
raw NeDBIT/DNA methylation UMAPs; after uses final PNAConv4 embeddings. Both
require the same five archived runs and preserve existing UMAP coordinates and
class colors. Missing runs or source figures cause an explicit skip warning.
Source paths are recorded under `accumulated_results/data/`. Install the PDF
composition dependency with `pip install -r train_gnn/requirements-plotting.txt`.
To generate only these composites, run
`python train_gnn/umap_composites.py --dataset <dataset-folder>`.

### XAI neighbor correlation and delta beta

`--stage xai` automatically creates a correlation bar plot and a scatter plot of
current-cohort Δβ versus external-cohort Δβ
(mean AML beta minus mean healthy-control beta in each cohort). Both use exactly the green
seed and blue other nodes displayed in the newly generated local XAI PDF.
The XAI node is a red star, and the dashed diagonal marks equal effects.
External effects come from the same CpG comparison table as compare-predictions. Numerical exports
include correlations, sample counts, group means, Δβ, and input provenance in
`generated/data/XAI_neighbor_correlations_node_<id>_<model>_pearson.tsv/json`.
Figures are written in PDF, PNG and SVG to the configured plot directory.
The standalone `train_gnn/plot_xai_neighbor_correlations.py` supports the same
plots without rerunning explanation.

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

### RN seed comparison

With `--stage xai --xai-group RN`, panel B compares the selected RN against
**every seed node** (`seed_score > 0`) using methylation profiles, including
seeds outside the displayed local subgraph. Ranked bars retain negative
correlations. Full CpG_Gene names, coefficients, and sample counts are exported
to `XAI_all_seed_correlations_node_<id>_<model>_pearson.tsv`; no seeds are
truncated. The RN composite contains only two panels: the explanation subgraph and
all-seed correlations. It requires no external cohort inputs. A center-only RN
graph is supported. LP retains its three-panel cohort-comparison layout.
