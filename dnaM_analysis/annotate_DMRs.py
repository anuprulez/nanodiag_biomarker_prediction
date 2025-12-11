import pandas as pd
import pyranges as pr
import numpy as np
import matplotlib.pyplot as plt


all_conditions = ["HT_smallEV", "Tspan8_smallEV", "Tspan8_largeEV", "HT1080_V_largeEV", "HT1080_V_smallEV"]
base_path = "data/"

'''
all combinations:


dmr_HT1080_V_largeEV_HT1080_V_smallEV
dmr_HT1080_V_largeEV_HT_smallEV
dmr_HT1080_V_largeEV_Tspan8_smallEV

dmr_HT1080_V_smallEV_HT_smallEV
dmr_HT1080_V_smallEV_Tspan8_smallEV

dmr_HT_smallEV_Tspan8_smallEV
dmr_Tspan8_largeEV_HT_smallEV
dmr_Tspan8_largeEV_Tspan8_smallEV

'''

conditions_first = [
    "HT1080_V_largeEV", 
    "HT1080_V_largeEV", 
    "HT1080_V_largeEV",  
    "HT1080_V_smallEV", 
    "HT1080_V_smallEV",
    "HT_smallEV",
    "Tspan8_largeEV",
    "Tspan8_largeEV"
]

conditions_second = [
    "HT1080_V_smallEV", 
    "HT_smallEV", 
    "Tspan8_smallEV", 
    "HT_smallEV",
    "Tspan8_smallEV",
    "Tspan8_smallEV",
    "HT_smallEV",
    "Tspan8_smallEV"
]


def assign_genes():

    for cond1, cond2 in zip(conditions_first, conditions_second):

        #cond1 = "HT1080_V_largeEV"
        #cond2 = "HT1080_V_smallEV"

        dmr_bedgraph = base_path + f"dmr_{cond1}_{cond2}.tabular"

        # Read as a generic table
        dmr_df = pd.read_csv(
            dmr_bedgraph,
            sep="\t",
            header=None,
            comment="#",
        )

        print(dmr_df.head())

        # filter by p-value threshold
        pvalue_threshold = 0.05
        dmr_df = dmr_df[dmr_df[6] <= pvalue_threshold]

        dmr_df_chr_str_end = dmr_df[[0, 1, 2, 4, 6, 7]]
        dmr_df_chr_str_end.columns = ["Chromosome", "Start", "End", "Methylation_Difference", "p_value", "q_value"]

        print("After p-value filtering:")
        print(dmr_df)

        print("For PyRanges:")
        print(dmr_df_chr_str_end)

        # Convert to PyRanges object
        dmrs = pr.PyRanges(dmr_df_chr_str_end)
        print(dmrs)

        gtf_path = base_path + "gencode.v38.annotation.gtf"   # replace with your actual file path

        # Read GTF into PyRanges
        genes_gtf = pr.read_gtf(gtf_path)

        # Keep only "gene" entries (one interval per gene)
        genes = genes_gtf[genes_gtf.Feature == "gene"]

        # Inspect columns to see where gene name / id lives
        print(genes)
        print(genes.columns)

        # Overlaps between DMRs (A) and genes (B)
        overlap = dmrs.join(genes)

        # Convert to a DataFrame
        overlap_df = overlap.df

        print("Overlapping DMRs and genes:")
        print(overlap_df)

        # Keep only useful columns
        # (Adjust "gene_name" / "gene_id" to whatever your GTF has)
        cols_to_keep = [
            "Chromosome", "Start", "End", "Score",   # DMR
            "gene_id", "gene_name", "Strand",         # Gene info
            "Methylation_Difference", "p_value", "q_value"  # DMR stats
        ]

        overlap_df = overlap_df[cols_to_keep]
        print(overlap_df.head())

        overlap_df = overlap_df.drop_duplicates(subset=["Chromosome", "Start", "End"])
        overlap_df.to_csv(f"{base_path}{cond1}_{cond2}_dmrs_p_value_corrected.tsv", sep="\t", index=False)
        plot_volcano(overlap_df, cond1, cond2)
        print(f"{cond1}_{cond2} done.")
        print("=========")


def plot_volcano(df, cond1, cond2):

    # Thresholds
    pval_threshold = 0.05
    meth_difference = 0.1

    df['color'] = 'gray'
    p_value_field = 'p_value'
    df.loc[(df[p_value_field] < pval_threshold) & (df['Methylation_Difference'] > meth_difference), 'color'] = 'red'    # upregulated
    df.loc[(df[p_value_field] < pval_threshold)& (df['Methylation_Difference'] < -meth_difference), 'color'] = 'blue'   # downregulated

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(df['Methylation_Difference'], df[p_value_field], c=df['color'], label=df["gene_name"], alpha=0.8, edgecolors='k')

    for i, row in df.iterrows():
        if row["color"] in ["red", "blue"]:
            plt.annotate(row["gene_name"], (row["Methylation_Difference"], row[p_value_field]), fontsize=12, alpha=0.8, rotation=90)

    # Add significance threshold lines
    plt.axhline(pval_threshold, color='black', linestyle='--', linewidth=1)
    plt.axvline(meth_difference, color='black', linestyle='--', linewidth=1)
    plt.axvline(-meth_difference, color='black', linestyle='--', linewidth=1)

    plt.xlabel('Methylation change')
    plt.ylabel('P-value (<0.05 significance threshold)')
    plt.title(f'Differential methylation vs P-value: {cond1 + "_" + cond2}')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(base_path + "volcano_plot_for_{}.png".format(cond1 + "_" + cond2), dpi=300)


assign_genes()