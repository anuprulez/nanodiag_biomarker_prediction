# Fixed-panel post-evaluation

Shared patient-level follow-up for the **33 breast-cancer CpGs** and **17 AML CpGs**.
The per-disease entry point is `evaluation/post_evaluation.py`; normally invoke it
through `run_pipeline.py --stage post-evaluation` from the disease directory.

## Run

```bash
python run_pipeline.py --stage post-evaluation \
  --folder data_corr_0.33_knn_n_3000_n_neg_10000 --models PNA
```

The fixed panels are reconstructed from
`accumulated_results/evaluation/likely_positive_common_unique_cpgs_PNA_<parameters>.tsv`:

- present among likely-positive predictions in all **five** runs;
- **|external Δβ| ≥ 0.30**, **external P < 0.01**, **external BH FDR < 0.01**;
- same nonzero direction in original and external cohorts.

The stage fails if this rule no longer yields exactly the 33 BC or 17 AML IDs
frozen in `resources/fixed_*_CpGs.tsv`. It
never silently replaces the panel, reruns training, or changes seeds. Run the
archived comparisons and accumulation first if those input tables are absent.
Only PNA is supported for these fixed panels. Without `--folder`, the runner
selects folders that already contain accumulated PNA evaluation tables; each
must satisfy the fixed-panel rule. The stage is **explicitly opt-in** and is not
included in `--stage all`, which can run before the five-run panel exists.

Useful options (also accepted by the pipeline):

```bash
# Inspect routing without writing any files.
python run_pipeline.py --stage post-evaluation --folder data_corr_0.33_knn_n_3000_n_neg_10000 --dry-run
# Use a supplied gene-symbol GMT instead of fetching Reactome.
python run_pipeline.py --stage post-evaluation --folder data_corr_0.33_knn_n_3000_n_neg_10000 --pathway-gmt /absolute/path/pathways.gmt
# Do not download pathway resources; other analyses still run.
python run_pipeline.py --stage post-evaluation --folder data_corr_0.33_knn_n_3000_n_neg_10000 --offline
# Change the descriptive matched-background comparison budget.
python run_pipeline.py --stage post-evaluation --folder data_corr_0.33_knn_n_3000_n_neg_10000 --random-panels 50 --background-size 1500
```

## Inputs and outputs

Reads the frozen dataset config's original methylation matrix, sample metadata
and seed list, the accumulated evaluation table, external raw matrices and
local platform annotation. CpG-gene rows are collapsed by CpG identity and
conflicting duplicated beta rows are rejected. Patient means are recomputed
from raw values and must reproduce the original/external evaluation effects.
Original metadata are joined by sample identifier, never by incidental row order.

Outputs go to `<dataset>/accumulated_results/post_evaluation/`:

| Output | Contents |
|---|---|
| `REPORT.md` | Results, model metrics, interpretation, limitations and artifact guide |
| `manifest.json` | Exact selected sites, source hashes, options, cohort sizes and method |
| `plots/delta_beta_forest.*` | Both cohorts' effects and pointwise 95% Welch confidence intervals |
| `plots/*patient_heatmap.*` | Raw beta heatmaps ordered by disease/study |
| `plots/*candidate_pca.*` | Descriptive patient PCA, separately within each cohort |
| `plots/external_cpg_distributions.*` | One publication figure containing all external candidate distributions |
| `plots/external_predicted_negative_cpg_distributions.*` | One external distribution figure for all 70 BC / 218 AML common predicted-negative CpGs |
| `plots/original_cpg_distributions_*.*` | Original-cohort patient beta distributions |
| `plots/*performance.*` | ROC and precision–recall (three original-CV / two displayed external panels) |
| `plots/panel_size_curve.*` | Performance for 1, 3, 5, 10 and all candidates |
| `plots/matched_random_panel_comparison.*` | Candidate versus mean/SD-matched random panels |
| `plots/genomic_annotation.*` | Chromosomes, gene-annotation counts, gene regions and CpG-island context |
| `plots/reactome_enrichment.*` | CpG-level pathway enrichment, when a GMT is available |
| `data/` | Panel, confidence intervals, patient matrices/annotations, predictions, folds, coefficients, performance/coverage, incremental AUROC, random-panel memberships and pathway tables |
| `cache/` | Requested patient matrices; reuse is checked against inputs and requested sites |

Plots are written as PNG and PDF. Performance curves, forest plots and the combined external
distribution figure also include editable SVG, 300-dpi PNG and embedded-font PDF.
Performance figures contain ROC and PR panels only; calibration is omitted.
The combined external distribution figure preserves all individual patient points
(rasterized in PDF), with vector text and boxplot elements. Pipeline stdout/stderr is captured in
`<dataset>/logs/post-evaluation.log`.

## Patient-level modeling

Use **five stratified patient folds**, seed `20261031`, identically for every
panel. The classifier is logistic regression with fixed `C=1` and `liblinear`;
there is no test-set-driven hyperparameter search. Median imputation and
standardization are fitted only on training patients. No graph-node split is
used as a patient split.

Compare seeds only, candidates only and their union, using CpGs measured in
both cohorts; `panel_coverage.tsv` records seed exclusions. Fit final models on
all original patients and transfer them without external refitting. Report
AUROC, average precision (AUPRC), Brier score and sensitivity at ≥95% specificity.
The external performance tables also include two additional disease-versus-healthy
classifiers trained on the original patients: **all common likely-positive CpGs**
(189 BC / 259 AML) and **all common predicted-negative CpGs** (70 BC / 218 AML).
These are the complete finite five-run group intersections from the accumulated
`likely_positive_common_unique_cpgs_*` and `negative_common_unique_cpgs_*` tables
underlying the threshold Δβ plots. The 0.30/P/FDR selection is **not applied** to
these two group panels. “Predicted negative” describes a CpG-node prediction,
not a patient diagnosis. The added panels remain in external transfer tables. The external ROC/PR figure
displays **Seeds only** and **Candidates only**; the original patient-CV plot
retains its three existing comparisons.
Exact membership lists, patient probabilities, bootstrap metrics and source
checksums are exported. The stage verifies the requested complete group counts
and fails rather than dropping missing sites silently.

ROC/PR baselines reflect the relevant cohort. Predicted probabilities reflect the sampled disease proportions, not a
population risk estimate.

Intervals use 200 stratified patient bootstraps of the already generated scores.
The incremental AUROC comparison uses 500 **paired** stratified bootstraps of
seed-only and combined OOF scores. These are conditional intervals: they do not
include model refitting, panel discovery or study-level uncertainty.

For the size curve, rank the fixed candidates by absolute original mean effect
**inside each training fold**, then evaluate the held-out patients. This avoids
ranking on the held-out patients, but does not undo initial panel-selection bias.

For random-panel comparisons, draw a reproducible pool of 800 externally shared
nonseed/noncandidate CpGs. Within each training fold, match each candidate to a
random choice among its 20 nearest available probes on overall patient mean beta
and SD (scaled across probes), without using disease labels. Match without
replacement within a panel. Evaluate 30 same-size panels on the same folds.
This is a descriptive background comparison, not an optimized or exhaustive
baseline and not a selection-adjusted hypothesis test. Memberships are saved.

## Biological annotation and pathways

Local platform annotations provide gene names, chromosomes and positions.
Gene-region and CpG-island context is extracted from the full local GEO
`aml_analysis/raw/GPL13534_family.soft.gz` platform table and cached in the
shared `resources/GPL13534_context.tsv.gz`. The parser stops at the end of the
platform table, before sample data. Multi-region mappings count a CpG once per
region and category counts can therefore exceed the panel size. If neither the
full source nor cached context exists, context analysis is marked unavailable.
Gene annotation does not establish a regulatory relationship.

The default pathway resource is the official Reactome GMT:
`https://reactome.org/download/current/ReactomePathways.gmt.zip`.
It is downloaded once under `<disease>/evaluation/post_evaluation_resources/`;
the actual GMT checksum is recorded. Use a fixed local GMT for exact resource
version reproducibility. Network errors or `--offline` without a cached GMT
produce an explicit analysis-status entry; they do not stop other analyses.

Pathway membership is mapped to **CpGs**, keeping multi-gene mappings, and
one-sided Fisher enrichment compares the selected panel against externally
shared, annotated CpGs. Pathways mapping to fewer than 10 or more than half of
background probes are excluded before testing. BH uses **all tested pathways**,
including zero-overlap pathways. A pathway covered by many probes therefore
has a larger background count; this avoids a naive gene-list enrichment.
This is a CpG-set exact test, **not missMethyl `gometh`**, and does not model
probe correlation or genomic-context matching. Interpret small-panel enrichment
as exploratory; nonsignificant results are retained and reported honestly.

## What these results can establish

These panels were selected using both original and external labels. Therefore
patient CV remains conditional on a previously selected feature set, and the
external cohort is a transfer stress test, **not untouched validation**.
Cases and controls in the external data come from different studies. The
original AML data also have study–disease confounding. Study cannot be adjusted
away when it completely predicts disease; the stage exports contingency tables
rather than fitting an unidentifiable adjusted disease effect.

The five graph-training runs measure algorithm stability, not five independent
patient replications. Heatmaps/PCA of selected sites and high AUC do not by
themselves establish clinical utility, disease specificity or independence
from batch effects. No survival/subtype or decision-curve result is fabricated
without appropriate outcomes and an intended clinical decision.

The next validation should freeze the panel and model and use new patients,
with cases and controls recruited within the same study and no feature
selection or model tuning on the final test patients.

## Dependencies and tests

Uses NumPy, pandas, SciPy, Matplotlib, scikit-learn and OmegaConf already in the
project's scientific environment. No R installation is required. Optional
pathway download uses Python's standard library. The first run streams several
GB of compressed raw data; subsequent runs reuse the requested-site cache.
Do not load pickle caches from untrusted third parties.

From `nanodiag_biomarker_prediction`:

```bash
python -m unittest post_evaluation_support.test_analysis
```

Per-disease routing tests are in `test_run_pipeline.py`.

## Three patient classifiers

`post-evaluation` now additionally trains **logistic regression, Random Forest
and XGBoost** for the same seed-only and fixed candidate-only panels. The external
performance PDF has four panels: seed ROC/PR on the first row and candidate ROC/PR
on the second row. Each contains three solid, distinctly colored model curves,
with no calibration panel or footer. The original three-model patient-CV comparison
is saved separately and does not replace the existing feature-panel CV figure.

If post-evaluation caches already exist, run the model comparison alone from
either disease directory:

```bash
python evaluation/compare_patient_classifiers.py   --config data_corr_0.33_knn_n_3000_n_neg_10000/config.yaml
```

This validates cached input fingerprints and reuses the exact feature lists.
All three models use identical five stratified original-patient folds. Within
each fold, median imputation uses training patients only; logistic regression
also fits its standardizer on training patients only. Final models fit all
original patients and then transfer unchanged to external patients. External
labels are used for scoring only, not fitting, tuning, early stopping or
threshold selection. The candidate-panel selection itself previously used
external labels, so it remains exploratory rather than untouched validation.

Prespecified model settings:

- Logistic regression: `C=1`, `solver=liblinear`, `max_iter=2000`.
- Random Forest: 500 trees, `min_samples_leaf=2`, `max_features=sqrt`, no class weighting.
- XGBoost: 500 trees, depth 3, learning rate 0.03, row/column subsampling 0.8,
  L2 regularization 1, histogram tree building, `scale_pos_weight=1`.
- Seed `20261031`; tree models use four CPU workers. There is no external-driven
  hyperparameter search or class rebalancing.

Outputs under `accumulated_results/post_evaluation/`:

- `CLASSIFIER_COMPARISON.md`: all model/panel/cohort results and interpretation.
- `classifier_comparison_manifest.json`: settings, versions and exact model feature order.
- `data/classifier_performance.tsv`: AUROC/AP/Brier, confidence intervals, sensitivity,
  specificity, accuracy, balanced accuracy and F1 at the fixed 0.50 threshold.
- `data/classifier_patient_predictions.tsv`: every patient's probability and predicted class.
- `data/classifier_pairwise_comparisons.tsv`: paired AUROC differences with bootstrap
  intervals, probability correlations and classification agreement.
- `plots/*classifier_comparison.*`: original-CV and external model-comparison ROC/PR figures.
- `plots/external_transfer_selected_cohort_performance.*`: updated external comparison.
- `plots/external_classifier_confusion_matrices.*`: counts and row percentages at 0.50.
- `models/*.joblib`: six fitted original-cohort preprocessing/classifier pipelines.

Original logistic-regression predictions must reproduce the existing saved
predictions before comparison outputs are accepted. Bootstrap intervals are
conditional on fixed predictions and feature selection; they do not incorporate
model refitting or study-level uncertainty. Study–disease confounding and the need
for an untouched cohort still apply. No model is chosen using external scores.

## Inverse-class-weighted comparisons

Run from either disease directory:

```bash
python evaluation/compare_patient_classifiers.py   --config data_corr_0.33_knn_n_3000_n_neg_10000/config.yaml   --inverse-class-weighting
```

This trains the same three models on the same seed/candidate panels with
**w_k = N_training / (2 × n_training_class_k)**, so the two training classes
have equal total weight and mean sample weight is 1. Weights use each fold's
training labels only. Final model weights use all original patients:

| Dataset | Healthy patients | Disease patients | Healthy weight | Disease weight |
|---|---:|---:|---:|---:|
| BC | 40 | 80 | 1.500000 | 0.750000 |
| AML | 464 | 62 | 0.566810 | 4.241935 |

The identical formula is passed as classifier `sample_weight` to logistic
regression, Random Forest and XGBoost. Built-in class weighting is not also
applied; XGBoost `scale_pos_weight=1` avoids double weighting. Median imputation
and logistic standardization still use training patients only and are unweighted.
No external labels, external prevalence or test-fold labels determine weights.

Outputs are under `post_evaluation/inverse_class_weighted/`, including saved
models, predictions, metrics, manifests and `CLASSIFIER_COMPARISON.md`.
`data/training_class_weights.tsv` records class counts, weights and equal total
class weight for every model, panel, CV training fold and final fit.

New separate figures (each has seed-only and candidate-only panels, with three
solid-color model curves) are exported as PDF/SVG/PNG:

- `plots/external_transfer_selected_cohort_classifier_ROC.*`
- `plots/external_transfer_selected_cohort_classifier_PRC.*`
- `plots/external_transfer_selected_cohort_classifier_confusion_matrices.*`
- The corresponding `original_patient_CV_classifier_*` plots.

The weighted combined ROC/PR figure is also saved in this subdirectory.
Full `--stage post-evaluation` now produces both the unweighted and weighted
comparisons. The original unweighted artifacts remain available for comparison.
Evaluation metrics and confusion counts use actual, unweighted patients; the PR
chance baseline remains the actual cohort disease proportion. Classification
threshold stays 0.50, fixed before external evaluation. Class weighting changes
the training objective and scores are not automatically calibrated to population
prevalence. Candidate-selection bias and study confounding still apply.

## Matching plot sets for direct comparison

Each disease now has explicit `unweighted/` and `inverse_class_weighted/` plot
sets under `post_evaluation/`. In each, `external_classifier_plot_set.pdf`
contains ROC, PRC and confusion matrices as three pages. Individual PDF/SVG/PNG
versions are saved in `plots/`, with exactly the same filenames and styling in
both folders. Titles explicitly state **No class weighting** or
**Inverse-frequency class weighting**. `PLOT_SET.md` records the corresponding
saved prediction and metric sources. `classifier_plot_sets.json` indexes the
sets. Rendering does not change or refit the saved classifiers.

## TabPFN3 and TabICLv2: five-model comparisons

Both weighted and unweighted sets also include **TabPFN3** and **TabICLv2**.
The exact local checkpoints used by the original-cohort analysis are reused and
SHA256-verified:

- `tabpfn-v3-classifier-v3_default.ckpt`
- `tabicl-classifier-v2-20260212.ckpt`

Estimator overrides (ensemble size, temperature and other preprocessing/ensemble
settings) come from each disease's frozen `config.yaml` model overrides. The
patient-fold random seed remains `20261031`, shared with the classical models.
Package versions, checkpoint hashes and every parameter are recorded in the
comparison manifest. The installed packages are TabPFN 8.0.0 and TabICL 2.2.0;
checkpoint/model generations are v3 and v2, respectively.

To add or refresh only the pretrained classifiers from validated caches:

```bash
python evaluation/compare_foundation_classifiers.py   --config data_corr_0.33_knn_n_3000_n_neg_10000/config.yaml
```

Both classical-model weight settings must already exist. Full post-evaluation
produces both and adds the foundation classifiers automatically. The standard
patient-comparison command also refreshes foundation results once both settings
exist. Foundation predictions are cached per panel/model using checkpoint,
parameter, feature, patient-matrix and fold fingerprints. Changing plotting
styles does not require repeating GPU inference. Existing classical predictions
are retained when foundation results are appended.

All models use the same patient folds and CpG panels. Foundation context consists
only of original training patients: each CV fold uses its training patients;
final external inference uses all original patients. The pretrained neural
weights are not fine-tuned on these cohorts. Median imputation is fit only on
the corresponding original training patients. External labels are used only
for evaluation, not context fitting or hyperparameter tuning.

**Weighting is model-specific and explicitly labeled.** The installed TabPFN3
and TabICLv2 `fit` APIs do not accept `sample_weight`. Their weighted versions
therefore use a post-ensemble training-prior correction:

`corrected_p[k] = p[k] * w[k] / sum_j(p[j] * w[j])`, with
`w[k] = N_training / (2*n_training_class_k)`.

CV correction uses each fold's training priors. External correction uses the
final original-cohort training priors. This is **not training-loss weighting**;
logistic regression, Random Forest and XGBoost retain their sample-weighted
training. No external class counts enter the correction. For a fixed binary
external cohort, constant prior correction preserves ranking, hence AUROC/AP,
but can change probabilities, Brier scores and the 0.50-threshold confusion
matrix. OOF rankings can differ because training priors vary between folds.

The five-model curves remain solid and distinctly colored. Confusion figures
have two feature-panel rows and five model columns. Every matched plot set
contains ROC, PRC and confusion pages. Predictions, metrics, all pairwise model
comparisons, reports and model artifacts are updated for both settings.
Foundation weighted artifacts wrap the fitted unweighted context and recorded
prior correction; keep their referenced base artifacts available.

The candidate-panel external-selection bias and study–disease confounding
continue to limit interpretation; these additions do not establish clinical
utility or untouched validation.

## Publication annotation figures

`genomic_annotation` and `reactome_enrichment` now export PDF, editable SVG and
300-dpi PNG through `annotation_publication.py`, used by the full pipeline.
Annotation bars show counts and percentages; overlapping gene-region categories
are explicit. Reactome displays the top 12 nonzero-overlap pathways using nominal
p-values, with exact CpG overlap and BH FDR alongside; all tested pathways still
enter BH correction. Neither current panel has significant Reactome enrichment.
`ANNOTATION_FINDINGS.md` in each disease's post-evaluation directory supplies
interpretation and figure captions. `annotation_figure_manifest.json` hashes the
saved inputs used for the current redraw. To redraw existing tables without
rerunning enrichment, use `python -m post_evaluation_support.annotation_publication
--out <post_evaluation_directory> --disease "Breast cancer"` (or `AML`).

## BC methylation–expression follow-up

The fixed 33 BC CpGs are evaluated against their legacy annotated genes using
public TCGA-BRCA matched methylation/RNA-seq, separately from GSE66695 (which
has no matched expression in its GEO record). Fetch only the selected rows:

```bash
python -m post_evaluation_support.fetch_bc_expression
python -m post_evaluation_support.methylation_expression --out breast_cancer_analysis/data_corr_0.33_knn_n_3000_n_neg_10000/accumulated_results/post_evaluation/methylation_expression
```

This is a separate follow-up command, not a model-training or panel-selection
stage. Exact sample identifiers are matched; ambiguous methylation-vial matches
are excluded. Tests are within primary tumors, with separate normal-tissue
results, Pearson sensitivity, age/PAM50 partial-rank sensitivity and subtype
analyses. Each test family receives BH correction. Same-complete-case raw
correlations accompany the adjusted results. Missing measurements are not
imputed. All 33 sites appear in the coverage audit, including four without usable
methylation values. HGNC documents the C7orf20→GET4 symbol update.

Outputs include `REPORT.md`, a source/hash manifest, complete association and
coverage tables, sample metadata and matched matrices, and PDF/SVG/300-dpi PNG
summary, scatterplot and adjustment figures. Source metadata and query responses
are cached. Associations are exploratory and do not establish methylation-driven
expression; purity, cell composition, copy number and batch remain unadjusted.

## BC tumor–normal expression and CpG direction concordance

`methylation_expression/direction_concordance/` contains the follow-up comparison
of gene expression logFC with original/external BC Δβ and paired TCGA Δβ.

```bash
python -m post_evaluation_support.fetch_bc_expression_genome
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 Rscript post_evaluation_support/bc_differential_expression.R breast_cancer_analysis/data_corr_0.33_knn_n_3000_n_neg_10000/accumulated_results/post_evaluation/methylation_expression
python -m post_evaluation_support.bc_direction_concordance --base breast_cancer_analysis/data_corr_0.33_knn_n_3000_n_neg_10000/accumulated_results/post_evaluation/methylation_expression
```

The primary limma model uses 77 paired patients and patient fixed effects;
variance moderation and BH use all expressed genes. The 775/84 full-cohort
sensitivity uses patient blocking, with within-patient correlation estimated
from the paired subset. Transformed RSEM expression is analyzed directly; it is
not supplied to DESeq2 or voom as raw counts. The promoter-only directional test
uses aligned gene/region annotations to identify the correct TSS-linked gene.
All-region summaries average available annotated-gene logFCs to give exactly one
point per CpG. Complete/unanimous mapping sensitivity, raw and adjusted p-values,
coverage, genome-wide and panel-only DE FDR, binomial intervals and feature-level
Spearman/Pearson correlations are retained. Sign tests do not require expression
significance; the sign expectation is primarily for promoter sites. Correlation
and binomial p-values across features are nominal because CpGs may co-vary.
The 33-site audit retains KCNG4 and MEG8, which fail the paired expression filter
but are covered by the larger-cohort sensitivity analysis.

## BC robust within-tissue analysis

The `methylation_expression/robust_followup/` outputs add paired-patient bootstrap
confidence intervals, published CPE/ABSOLUTE purity, target-gene continuous GISTIC2
copy number, and an hg19 transcript-based ±250-kb cis target search. No pooled
unadjusted tumor/normal correlation is calculated. Original log2(RSEM+1) expression
and beta values are analyzed with within-tissue Spearman; partial rank models
control purity/CN, with ABSOLUTE and age/PAM50 sensitivity. Adjusted CIs are
explicitly approximate Fisher-z intervals; raw CIs use patient bootstrap.

```bash
python -m post_evaluation_support.fetch_bc_robust_sources
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m post_evaluation_support.bc_robust_associations --base breast_cancer_analysis/data_corr_0.33_knn_n_3000_n_neg_10000/accumulated_results/post_evaluation/methylation_expression
python -m post_evaluation_support.bc_updated_priority_plot --base breast_cancer_analysis/data_corr_0.33_knn_n_3000_n_neg_10000/accumulated_results/post_evaluation/methylation_expression
```

This requires the cached whole-expression matrix from the preceding limma
follow-up. Purity data are the published Aran et al estimates distributed in
TCGAbiolinks (R data conversion uses the task's local R environment). Source
URLs, query caches and input hashes are retained. HGNC approved/previous-symbol
mappings and hg19 refGene TSSs are used, with coordinate conversion documented.
Duplicate annotated/cis targets are collapsed; BH includes every testable target
in the union, separately per tissue/model, with additional joint FDR columns.
The separate paired and patient-blocked limma tumor/normal expression differences
are retained as their own tables. No enhancer contacts, full cell-type
composition, technical-batch correction or new matched GEO validation is claimed.

The updated priority plot retains all 33 CpGs but highlights three exploratory
follow-up candidates meeting inverse group changes, negative adjusted tumor
associations under all three models, expression FDR and a defined TSS proximity
window. The exact conditions are in `REPORT.md` and
`data/updated_functional_followup_priorities.tsv`. These are candidate regulatory
links, not experimentally validated functional CpGs.

## Explicit BC cross-cohort comparison

`python -m post_evaluation_support.bc_cross_cohort_comparison --base <methylation_expression_directory>`
creates `cross_cohort_comparison/`: original GSE66695 Δβ versus TCGA-BRCA limma
expression log₂FC, with one point per CpG, gene-level DE significance, mapping
flags, sign tests and feature-level correlations. This reuses validated saved
methylation and limma effects. It does not pair patients across cohorts or
calculate a pooled sample-level correlation. The plot has no arrows or genomic
region grouping. Multi-gene averages and promoter-specific targets are explicitly
distinguished in the tables and report.

The full 189 common likely-positive BC CpGs can be compared using
`python -m post_evaluation_support.bc_likely_positive_cross_cohort --base <methylation_expression_directory>`.
This writes `likely_positive_cross_cohort_comparison/`, without applying the
33-site effect/significance selection thresholds. Every candidate is retained
in the coverage table. The script reuses genome-wide limma DE estimates,
provides per-gene and one-point-per-CpG results, promoter-target and strict-mapping
sensitivities, and verifies that the previous selected 33 points reproduce
exactly. It does not fit new within-tumor correlation models for all 189 sites.

Cross-cohort scatterplots now label every evaluable point outside the
both-negative quadrant with its CpG ID and the expression gene(s) used. Points
with both negative values remain gray and unlabeled. `point_labels.py` uses deterministic collision avoidance, with labels
constrained to their point's quadrant and no arrows. `displayed_CpG_gene_labels.tsv`
records the exact labeled points. The likely-positive plot is enlarged for its
70 labels; the selected-panel plot labels 15 points. Genomic annotation data
and statistical results are unchanged by these display rules.

## Cross-cohort methylation–expression plots (BC and AML)

The `post-evaluation` stage now runs the complete expression download-to-plot
workflow automatically, after the existing patient/classifier analyses. To run
just these analyses through the same stage (from the repository root):

```bash
python breast_cancer_analysis/run_pipeline.py --stage post-evaluation \
  --folder data_corr_0.33_knn_n_3000_n_neg_10000 \
  --post-evaluation-analysis cross-cohort
python aml_analysis/run_pipeline.py --stage post-evaluation \
  --folder data_corr_0.33_knn_n_3000_n_neg_10000 \
  --post-evaluation-analysis cross-cohort
```

`--folder` follows the existing pipeline convention: relative folder names are
resolved under the disease directory, or an absolute path can be supplied.
The frozen five-run panels and original training methylation matrix/metadata
must already exist (produced by the upstream pipeline). This stage does not
reselect CpGs or substitute external methylation deltas. It prepares missing
fixed-panel annotation and sample-level methylation inputs, verifies the
original group means, downloads missing expression resources, normalizes and
fits expression models, renders PDF/SVG/300-dpi PNG plots, and validates results.
There are no manual download or R commands between steps.

Dependencies: use the existing model environment with Python numpy, pandas,
scipy >= 1.11, matplotlib, scikit-learn, requests and OmegaConf, plus R with
limma and statmod. A standalone analysis environment can be created with:

```bash
conda env create -f post_evaluation_support/environment-cross-cohort.yml
conda activate nanodiag-cross-cohort
```

R discovery checks `--rscript`, `RSCRIPT`, `PATH`, then an existing ancestor's
`.analysis_envs/bc-limma/bin/Rscript`. Missing R packages fail explicitly; the
stage does not silently replace limma with a different statistical method.
Use `--rscript /path/to/Rscript` for another installation.

Flags (also accepted by both `evaluation/post_evaluation.py` wrappers):

- `--post-evaluation-analysis all` (default): existing post-evaluation plus the
  new cross-cohort workflow. `cross-cohort` skips unrelated classifier training.
- `--refresh-expression`: rebuild normalized matrices and rerun limma while
  reusing valid downloaded source files. Downloads use temporary files and
  atomic replacement, so an interrupted transfer cannot become a valid cache.
- `--offline`: prohibit new resource/query requests. Missing or incompatible
  source caches produce a clear error; rebuilding from complete caches works.
- Normal reruns reuse preprocessing/limma only when both input and output
  content hashes still match. Plots and numerical validation are always run.
  After upgrading R/limma or changing the analysis environment, use
  `--refresh-expression` to refit with that environment.

BC uses TCGA-BRCA HiSeqV2 log2(RSEM normalized count + 1) from UCSC Xena. It
reconstructs the previous 775-tumor/84-normal expression set by exact sample-type
matching to GDC methylation IDs, fetches the expressed genome in resumable,
query-validated chunks, and fits the paired and full patient-blocked limma
models. The final X values use the full patient-blocked model; Y remains the
original GSE66695 80-tumor/40-normal delta beta. HGNC GET4 mapping is retained.
No purity/copy-number follow-up or matched patient-correlation result is
required as a pre-existing cache to produce the cross-cohort plot.

AML downloads GSE15061 and the GPL570 annotation independently of BC, plus an
HGNC snapshot. Only the DS representation of each biological sample is used:
202 AML and 69 non-leukemic marrow samples; MDS and duplicate DQN3 records are
excluded. DS signals are log2 transformed, quantile normalized, and averaged
across probes with unambiguous gene mappings. Limma fits group + study center
and an unadjusted sensitivity model. Y remains the original GSE58477 AML62
minus GSE50660 control464 delta beta. PRHOXNB/URAD lacks an unambiguous probe,
so its missing expression stays missing: 16 of the 17 fixed sites are plotted.
GPL570 download reads the complete platform table and stops before the huge
following sample family. It never downloads all family sample records.

Outputs within each selected dataset:

```text
accumulated_results/post_evaluation/
  data/fixed_candidate_panel.tsv
  data/candidate_genomic_annotation.tsv
  data/original_candidate_beta.tsv.gz
  methylation_expression/
    pipeline_manifest.json
    cross_cohort_comparison/
      REPORT.md
      manifest.json
      validation.json
      data/                         # CpG, gene, significance and sensitivity tables
      plots/                        # PDF, editable SVG, 300-dpi PNG
    direction_concordance/          # BC source chunks, genome-wide limma + models
    source/                         # BC query caches and source metadata
```

AML's raw source, normalized matrices and genome-wide limma tables are inside
`cross_cohort_comparison/source` and `cross_cohort_comparison/data`. Logs remain
at `<dataset>/logs/post-evaluation.log`.

Main scripts: `cross_cohort_pipeline.py` orchestrates; `expression_resources.py`
handles downloads and R discovery; `fetch_bc_expression.py` and
`fetch_bc_expression_genome.py` fetch BC inputs; `bc_differential_expression.R`
and `aml_expression_limma.R` fit limma; `bc_direction_concordance.py`,
`bc_cross_cohort_comparison.py` and `aml_cross_cohort_comparison.py` produce the
summaries/figures. The two `validate_*_cross_cohort.py` scripts independently
check methylation means, frozen identities, label selection, genome-wide FDR
and model coefficients (numpy GLS for BC patient blocks, OLS for AML centers).

The plots retain the selected layout: centered X=0, enlarged fonts, promoter /
gene-body symbols, labels only for inverse-sign candidates and arrows pointing
to their actual coordinates. They compare effects across CpGs in independent
cohorts, not methylation/expression correlations across unrelated patients.
AML methylation disease/study confounding and expression center sensitivity
remain explicit in the generated report; automation does not remove them.


## Candidate-panel expression publication figures

`expression_publication.py` runs automatically after the cross-cohort analysis in
`--stage post-evaluation`, including `--post-evaluation-analysis cross-cohort`.
It writes `methylation_expression/publication_figures/` inside post-evaluation.
These are selected-panel adaptations of Liu et al. (2020), Figures 2, 3, 4B/C and
6: context distributions, observed effect maps, original/external β heatmaps and
annotation/effect locus maps. It does not invent matched-sample correlations,
pan-cancer comparisons or ATAC/ChIP tracks. The generated README documents the
explicit differences from the reference, and the manifest records input/script
hashes. Main scatterplots omit correlation annotations and use Δβ in their titles.

## Complete candidate CpG tables

`post_evaluation_support/candidate_tables.py` exports exhaustive tables for all 17 AML and 33 BC frozen candidate CpGs. No CpGs are removed for missing expression or annotation, and multi-gene sites remain one row per CpG. Exports are automatically refreshed by the `all` and `cross-cohort` post-evaluation analyses. To regenerate only these tables from existing saved inputs, run from the project directory containing the disease folders:

```bash
python aml_analysis/run_pipeline.py --stage post-evaluation --folder data_corr_0.33_knn_n_3000_n_neg_10000 --post-evaluation-analysis candidate-tables --offline
python breast_cancer_analysis/run_pipeline.py --stage post-evaluation --folder data_corr_0.33_knn_n_3000_n_neg_10000 --post-evaluation-analysis candidate-tables --offline
```

Outputs are in each dataset's `accumulated_results/post_evaluation/data/`: `AML_17_CpG_genomic_context.{tsv,csv,md,json}` or `BC_33_CpG_genomic_context.{tsv,csv,md,json}`. CSV and TSV retain full numeric precision; Markdown displays six decimal places. JSON records input hashes and source columns. These exports require existing `fixed_candidate_panel.tsv` and `candidate_genomic_annotation.tsv`; missing or mismatched inputs fail explicitly.

Columns: `CpG_site`, `Gene`, `Delta_beta_external`, `Delta_beta_original`, `Gene_locations`, `Island_locations`, `Chromosome`, and `CpG_coordinate_MAPINFO`. CSV/TSV additionally preserve `Gene_annotation_raw` and `Gene_locations_raw`, including repeated transcript entries. Display columns deduplicate semicolon-separated genes/regions without discarding unique memberships; their list positions should not be interpreted as gene-region pairs. Coordinates are CpG probe coordinates copied from the legacy GPL13534 annotation, not gene boundaries or newly mapped positions.

Both delta-beta columns mean disease mean beta minus healthy/control mean beta, on the beta scale (positive = hypermethylated). Original AML values use the original cohort; external AML values compare GSE62298 AML with GSE40279 healthy controls. Original BC values use GSE66695; external BC values compare GSE226569 BC with GSE101961 controls. External cases and controls come from separate studies, so these differences can include study effects.

`Gene_locations` retains all annotated gene-relative regions (TSS1500, TSS200, 5'UTR, 1stExon, Body, 3'UTR as available). Missing genes/regions are labeled `Unannotated`; missing island context is labeled `Unspecified / open sea`, without claiming that a blank annotation proves open-sea membership. Gene symbols are the original annotation symbols. No expression filtering, new candidate selection, enrichment testing, or gene-symbol remapping is performed.
