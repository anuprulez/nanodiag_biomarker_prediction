# PNA: Original breast tissue versus external breast tissue

Exploratory comparison of independent breast-tissue cohorts; matching platform does not remove study effects

Δβ = mean(BC beta) − mean(healthy beta). ΔΔβ = external Δβ − original Δβ.
Original GSE66695 cohort (breast tumors and normal breast tissue, GPL13534 (450K)): 40 healthy controls and 80 BC patients. External breast tissue cohorts (GPL13534): GSE101961 cancer-free normal breast and GSE226569 FFPE breast tumors.
CpGs are matched by exact CpG_ID. Multiple gene mappings count once per CpG in plots and summary statistics.

| Prediction group | Input CpG-gene rows | Input unique CpGs | Common unique CpGs | Pearson r | Median absolute external Δβ |
|---|---:|---:|---:|---:|---:|
| Predicted negative | 959 | 832 | 307 | 0.8955 | 0.1165 |
| Predicted likely positive | 510 | 480 | 240 | 0.7948 | 0.2094 |

Common CpGs assigned to both prediction groups: 2. Such CpGs remain in both group panels; the groups are not assumed independent.

Plots: `delta_beta_original_vs_external_PNA_corr_0.33_knn_n_3000_n_neg_10000.png` and `delta_beta_shift_by_prediction_group_PNA_corr_0.33_knn_n_3000_n_neg_10000.png`, with PDF versions.
Tables retain all extracted features, all matching CpG-gene rows, unique matched CpGs, and unmatched predictions (Matched_external_CpG=false).

Breast tissue effects are compared across separate studies and processing protocols. Tissue composition, study, processing and tumor purity can change effect direction and size. No batch, demographic or cell-composition adjustment. Propagated priority classes are not ground-truth differential methylation labels; internal classification accuracy does not establish biological replication.

Regenerate from breast_cancer_analysis with `python evaluation/compare_predictions.py --model PNA --data /home/ubuntu/data/anup/nanodiag-biomarker/test-train/nanodiag_biomarker_prediction/breast_cancer_analysis/data_corr_0.33_knn_n_3000_n_neg_10000`.
