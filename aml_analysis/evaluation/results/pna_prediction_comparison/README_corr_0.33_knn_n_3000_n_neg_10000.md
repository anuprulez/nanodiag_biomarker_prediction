# PNA prediction groups versus external cohorts

Δβ = mean(AML beta) − mean(healthy beta). ΔΔβ = external Δβ − original Δβ.
Original cohort: 464 healthy controls and 62 AML samples. External cohort: GSE40279 healthy controls and GSE62298 AML.
CpGs are matched by exact CpG_ID. Multiple gene mappings count once per CpG in plots and summary statistics.

| Prediction group | Input CpG-gene rows | Input unique CpGs | Common unique CpGs | Pearson r | Median absolute external Δβ |
|---|---:|---:|---:|---:|---:|
| Predicted negative | 908 | 825 | 824 | 0.8898 | 0.0501 |
| Predicted likely positive | 340 | 315 | 315 | 0.8956 | 0.1178 |

Common CpGs assigned to both prediction groups: 0. Such CpGs remain in both group panels; the groups are not assumed independent.

Plots: `delta_beta_original_vs_external_corr_0.33_knn_n_3000_n_neg_10000.png` and `delta_beta_shift_by_prediction_group_corr_0.33_knn_n_3000_n_neg_10000.png`, with PDF versions.
Tables retain all extracted features, all matching CpG-gene rows, unique matched CpGs, and unmatched predictions (Matched_external_CpG=false).

Descriptive cross-study effects; no adjustment for batch, age, sex or cell composition; predicted classes are model outputs, not external ground truth.

Regenerate from aml_analysis with `python evaluation/compare_pna_predictions.py --data /home/ubuntu/data/anup/nanodiag-biomarker/test-train/nanodiag_biomarker_prediction/aml_analysis/data_corr_0.33_knn_n_3000_n_neg_10000`.
