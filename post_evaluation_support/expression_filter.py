"""Gene-level expression selection for cross-cohort effect plots."""
EXPRESSION_P_CUTOFF = 0.01
EXPRESSION_FDR_CUTOFF = 0.01


def expression_selected(p_value, fdr):
    """Require both raw p and existing genome-wide BH FDR, strictly below cutoffs."""
    return (p_value < EXPRESSION_P_CUTOFF) & (fdr < EXPRESSION_FDR_CUTOFF)
