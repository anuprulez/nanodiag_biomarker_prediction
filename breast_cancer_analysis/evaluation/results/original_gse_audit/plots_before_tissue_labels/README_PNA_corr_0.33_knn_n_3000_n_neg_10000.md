# PNA prediction groups versus external cohorts

Δβ = mean(BC beta) − mean(healthy beta). ΔΔβ = external Δβ − original Δβ.
Original cohort: 30 healthy controls and 50 BC samples. External cohort: GSE101961 healthy controls and GSE226569 BC.
CpGs are matched by exact CpG_ID. Multiple gene mappings count once per CpG in plots and summary statistics.

| Prediction group | Input CpG-gene rows | Input unique CpGs | Common unique CpGs | Pearson r | Median absolute external Δβ |
|---|---:|---:|---:|---:|---:|
| Predicted negative | 926 | 859 | 227 | -0.3631 | 0.0878 |
| Predicted likely positive | 524 | 499 | 123 | 0.0704 | 0.0773 |

Common CpGs assigned to both prediction groups: 0. Such CpGs remain in both group panels; the groups are not assumed independent.

Plots: `delta_beta_original_vs_external_PNA_corr_0.33_knn_n_3000_n_neg_10000.png` and `delta_beta_shift_by_prediction_group_PNA_corr_0.33_knn_n_3000_n_neg_10000.png`, with PDF versions.
Tables retain all extracted features, all matching CpG-gene rows, unique matched CpGs, and unmatched predictions (Matched_external_CpG=false).

Descriptive cross-study effects; no adjustment for batch, age, sex or cell composition; predicted classes are model outputs, not external ground truth.

Regenerate from breast_cancer_analysis with `python evaluation/compare_predictions.py --model PNA --data /home/ubuntu/data/anup/nanodiag-biomarker/test-train/nanodiag_biomarker_prediction/breast_cancer_analysis/data_corr_0.33_knn_n_3000_n_neg_10000`.
