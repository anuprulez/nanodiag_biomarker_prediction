import pandas as pd
import pyranges as pr
import numpy as np
import matplotlib.pyplot as plt
import os
import re

all_conditions = ["HT_smallEV", "Tspan8_smallEV", "Tspan8_largeEV", "HT1080_V_largeEV", "HT1080_V_smallEV"]
coverage = 5
pvalue_threshold = 0.05
gene_annotation_path = "data/"
base_path = f"data/coverage_{coverage}/"

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



import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def canonical_chr(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    s = re.sub(r"^chr", "", s, flags=re.IGNORECASE).strip()
    if s.upper() in {"M", "MT"}:
        return "MT"
    if s.isdigit():
        n = int(s)
        if 1 <= n <= 22:
            return str(n)
        return None
    if s.upper() in {"X", "Y"}:
        return s.upper()
    return None


def chr_to_numeric(c):
    """Fixed x-position for chromosome: 1..22, X=23, Y=24, MT=25."""
    if c is None:
        return np.nan
    if c.isdigit():
        n = int(c)
        return n if 1 <= n <= 22 else np.nan
    if c == "X":
        return 23
    if c == "Y":
        return 24
    if c == "MT":
        return 25
    return np.nan


def add_fixed_chr_axis_and_jitter(df, chr_col="Chromosome", jitter=0.28, seed=1):
    """
    Adds:
      - CHR (canonical)
      - CHR_NUM (fixed 1..25)
      - X = CHR_NUM + jitter
    """
    d = df.copy()
    d["CHR"] = d[chr_col].map(canonical_chr)
    d = d.dropna(subset=["CHR"]).copy()

    d["CHR_NUM"] = d["CHR"].map(chr_to_numeric)
    d = d.dropna(subset=["CHR_NUM"]).copy()

    rng = np.random.default_rng(seed)
    d["X"] = d["CHR_NUM"].astype(float) + rng.uniform(-jitter, jitter, size=len(d))
    return d


def annotate_points(ax, df, x_col="X", y_col="Methylation_Difference",
                    start_col="Start", end_col="End", fontsize=7,
                    xytext=(2, 2), max_labels=None):
    """Annotate each point with 'Start-End'."""
    if start_col not in df.columns or end_col not in df.columns:
        print("[WARN] Start/End columns missing; skipping annotations.")
        return

    d = df.copy()
    d[start_col] = pd.to_numeric(d[start_col], errors="coerce")
    d[end_col] = pd.to_numeric(d[end_col], errors="coerce")
    d = d.dropna(subset=[x_col, y_col, start_col, end_col])

    if max_labels is not None:
        d = d.head(max_labels)

    for _, r in d.iterrows():
        label = f"{int(r[start_col])}-{int(r[end_col])}"
        ax.annotate(
            label,
            (r[x_col], r[y_col]),
            textcoords="offset points",
            xytext=xytext,
            ha="left",
            va="bottom",
            fontsize=fontsize,
        )


def plot_chr_methylation_change(
    dmr_df,
    dmp_df=None,
    outpath="chr_meth_change.png",
    title="DMRs and DMPs by chromosome",
    p_col_dmr="p_value_MWU",
    p_col_dmp="p_value",
    effect_col="Methylation_Difference",
    p_threshold=0.05,
    effect_threshold=0.10,
    jitter=0.28,
    dpi=300,
    annotate=True,
    annotate_only_significant=False,
    annotate_max_labels=None,
):
    fig, ax = plt.subplots(figsize=(14, 5))

    # ---- DMR prep
    dmr = dmr_df.copy()
    dmr[effect_col] = pd.to_numeric(dmr[effect_col], errors="coerce")
    dmr[p_col_dmr] = pd.to_numeric(dmr[p_col_dmr], errors="coerce")
    dmr = dmr.dropna(subset=[effect_col, p_col_dmr]).copy()

    #dmr = add_fixed_chr_axis_and_jitter(dmr, chr_col="Chromosome", jitter=jitter, seed=1)

    dmr = add_fixed_chr_axis_and_spread(
        dmr,
        chr_col="Chromosome",
        start_col="Start",
        end_col="End",
        jitter=jitter
    )

    # ---- DMP prep (optional, SAME coordinate system)
    dmp = None
    if dmp_df is not None and len(dmp_df) > 0:
        dmp = dmp_df.copy()
        dmp[effect_col] = pd.to_numeric(dmp[effect_col], errors="coerce")
        dmp[p_col_dmp] = pd.to_numeric(dmp[p_col_dmp], errors="coerce")
        dmp = dmp.dropna(subset=[effect_col, p_col_dmp]).copy()
        dmp = add_fixed_chr_axis_and_jitter(dmp, chr_col="Chromosome", jitter=jitter, seed=2)

    # ---- Significance split (use <= and > to avoid losing p==threshold)
    dmr_sig = dmr[dmr[p_col_dmr] <= p_threshold].copy()
    dmr_nsig = dmr[dmr[p_col_dmr] > p_threshold].copy()

    if dmp is not None:
        dmp_sig = dmp[dmp[p_col_dmp] <= p_threshold].copy()
        dmp_nsig = dmp[dmp[p_col_dmp] > p_threshold].copy()

    # ---- Background bands: MUST match fixed chromosome positions (1..25)
    for chrnum in range(1, 26):
        if chrnum % 2 == 0:
            ax.axvspan(chrnum - 0.5, chrnum + 0.5, alpha=0.06)

    # ---- Scatter
    ax.scatter(dmr_nsig["X"], dmr_nsig[effect_col], s=12, alpha=0.25, edgecolors="none", label="DMR (p>thr)")
    ax.scatter(dmr_sig["X"],  dmr_sig[effect_col],  s=28, alpha=0.90, edgecolors="none", label="DMR (p≤thr)")

    if dmp is not None:
        ax.scatter(dmp_nsig["X"], dmp_nsig[effect_col], s=10, alpha=0.20, marker="x", label="DMP (p>thr)")
        ax.scatter(dmp_sig["X"],  dmp_sig[effect_col],  s=22, alpha=0.90, marker="x", label="DMP (p≤thr)")

    # ---- Threshold lines
    if effect_threshold is not None:
        ax.axhline(+effect_threshold, linestyle="--", linewidth=1)
        ax.axhline(-effect_threshold, linestyle="--", linewidth=1)
    ax.axhline(0, linewidth=1, alpha=0.6)

    # ---- Axes: fixed ticks
    xticks = list(range(1, 23)) + [23, 24, 25]
    xlabels = [str(i) for i in range(1, 23)] + ["X", "Y", "MT"]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    ax.set_xlim(0.5, 25.5)

    ax.set_title(title)
    ax.set_xlabel("Chromosome")
    ax.set_ylabel("Methylation change")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.01)

    ax.legend(frameon=False, ncol=2)

    # ---- Annotations
    if annotate:
        if annotate_only_significant:
            annotate_points(ax, dmr_sig, y_col=effect_col, max_labels=annotate_max_labels)
            if dmp is not None:
                annotate_points(ax, dmp_sig, y_col=effect_col, max_labels=annotate_max_labels)
        else:
            annotate_points(ax, dmr, y_col=effect_col, max_labels=annotate_max_labels)
            if dmp is not None:
                annotate_points(ax, dmp, y_col=effect_col, max_labels=annotate_max_labels)

    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    plt.savefig(outpath, dpi=dpi)
    plt.close(fig)
    print(f"[OK] Saved: {outpath}")

def add_fixed_chr_axis_and_spread(
    df,
    chr_col="Chromosome",
    start_col="Start",
    end_col="End",
    jitter=0.28,
):
    """
    Fixed chromosome axis (1..22,X=23,Y=24,MT=25) + deterministic symmetric spread per chromosome.

    Key property:
      - if a chromosome has exactly 1 point => X == CHR_NUM (no shift)
      - if multiple points on same chr => evenly spaced offsets in [-jitter, +jitter]
    """
    d = df.copy()
    d["CHR"] = d[chr_col].map(canonical_chr)
    d = d.dropna(subset=["CHR"]).copy()

    d["CHR_NUM"] = d["CHR"].map(chr_to_numeric)
    d = d.dropna(subset=["CHR_NUM"]).copy()

    # Use midpoint for ordering (or just Start if you prefer)
    d[start_col] = pd.to_numeric(d[start_col], errors="coerce")
    if end_col in d.columns:
        d[end_col] = pd.to_numeric(d[end_col], errors="coerce")
        d["POS"] = (d[start_col] + d[end_col]) / 2.0
    else:
        d["POS"] = d[start_col]

    d = d.dropna(subset=["POS"]).copy()

    # Deterministic symmetric offsets per chromosome
    d["X"] = d["CHR_NUM"].astype(float)  # start exactly at the tick
    for chrnum, idx in d.groupby("CHR_NUM").groups.items():
        sub = d.loc[idx].sort_values("POS")
        n = len(sub)
        if n == 1:
            offsets = np.array([0.0])  # IMPORTANT: no shift
        else:
            offsets = np.linspace(-jitter, jitter, n)  # symmetric around 0
        d.loc[sub.index, "X"] = float(chrnum) + offsets

    return d



def create_manhattan_plot():

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

        print("Initial DMR DataFrame:")
        print(dmr_df.head())

        # Metilene output columns: http://legacy.bioinf.uni-leipzig.de/Software/metilene/manual.pdf

        dmr_df.columns = ["Chromosome", "Start", "End", "q-value", \
                          "Methylation_Difference", "Num_CpGs", "p_value_MWU", \
                            "p_value_2D_KS", "Methylation_1", "Methylation_2"]

        # filter by p-value threshold

        dmr_df = dmr_df[dmr_df["p_value_MWU"] <= pvalue_threshold]
        print("After p-value filtering:")
        print(dmr_df)

        #dmr_df_chr_str_end = dmr_df[[0, 1, 2, 4, 5, 6, 7]]
        #dmr_df_chr_str_end.columns = ["Chromosome", "Start", "End", "Methylation_Difference", "p_value", "q_value"]
        dmr_df_chr_str_end = dmr_df[["Chromosome", "Start", "End", "Methylation_Difference", "Num_CpGs", "p_value_MWU"]]
        print("Useful columns for Manhattan plot:")
        print(dmr_df_chr_str_end)

        # ---- Manhattan plot (NEW) ----
        outpath = os.path.join(base_path, f"chr_effect_{cond1}_vs_{cond2}.png")
        title = f"DMRs/DMPs by chromosome: {cond1} vs {cond2}"
        effect_threshold = 0.10

        plot_chr_methylation_change(
            dmr_df=dmr_df_chr_str_end,
            dmp_df=None,
            outpath=outpath,
            title=title,
            p_threshold=pvalue_threshold,
            effect_threshold=effect_threshold,
        )

        #break


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

        print("Initial DMR DataFrame:")
        print(dmr_df.head())

        # Metilene output columns: http://legacy.bioinf.uni-leipzig.de/Software/metilene/manual.pdf

        dmr_df.columns = ["Chromosome", "Start", "End", "q-value", \
                          "Methylation_Difference", "Num_CpGs", "p_value_MWU", \
                            "p_value_2D_KS", "Methylation_1", "Methylation_2"]

        # filter by p-value threshold

        dmr_df = dmr_df[dmr_df["p_value_MWU"] <= pvalue_threshold]
        print("After p-value filtering:")
        print(dmr_df)

        #dmr_df_chr_str_end = dmr_df[[0, 1, 2, 4, 5, 6, 7]]
        #dmr_df_chr_str_end.columns = ["Chromosome", "Start", "End", "Methylation_Difference", "p_value", "q_value"]
        dmr_df_chr_str_end = dmr_df[["Chromosome", "Start", "End", "Methylation_Difference", "Num_CpGs", "p_value_MWU"]]

        print("For PyRanges:")
        print(dmr_df_chr_str_end)

        # Convert to PyRanges object
        dmrs = pr.PyRanges(dmr_df_chr_str_end)
        print(dmrs)

        ## Download HG38 GTF 
        # https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_38/gencode.v38.annotation.gtf.gz

        gtf_path = gene_annotation_path + "gencode.v38.annotation.gtf" 

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
            "Methylation_Difference", "Num_CpGs", "p_value_MWU"  # DMR stats
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
    p_value_field = 'p_value_MWU'
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


#assign_genes()

create_manhattan_plot()