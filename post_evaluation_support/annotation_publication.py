"""Reproducible annotation figures from saved tables; no refitting or enrichment rerun."""
from pathlib import Path
import argparse
import json
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from .publication import save_publication

STYLE = {'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 12,
         'axes.labelsize': 11, 'pdf.fonttype': 42, 'svg.fonttype': 'none'}


def annotation_figure(selected, disease):
    n = len(selected)
    genes = next(c for c in ('UCSC_RefGene_Name', 'Genes', 'Gene') if c in selected)
    gene_counts = selected[genes].fillna('').map(lambda x: len({g.strip() for g in x.split(';') if g.strip()}))
    chrom = selected.CHR.fillna('Unknown').astype(str).value_counts()
    chrom = chrom.reindex(sorted(chrom.index, key=lambda x: (0, int(x)) if x.isdigit() else (1, x)))
    regions = selected.get('UCSC_RefGene_Group', pd.Series('', index=selected.index)).fillna('').map(
        lambda x: sorted({v for v in x.split(';') if v}) or ['Unannotated']).explode().value_counts()
    order = ['TSS1500', 'TSS200', "5'UTR", '1stExon', 'Body', "3'UTR", 'Unannotated']
    regions = regions.reindex([v for v in order if v in regions] + [v for v in regions.index if v not in order])
    islands = selected.get('Relation_to_UCSC_CpG_Island', pd.Series('', index=selected.index)).fillna('Unannotated / open sea').replace('', 'Unannotated / open sea').value_counts()
    order = ['Island', 'N_Shore', 'S_Shore', 'N_Shelf', 'S_Shelf', 'Unannotated / open sea']
    islands = islands.reindex([v for v in order if v in islands] + [v for v in islands.index if v not in order])
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.2))
        specs = [(chrom, 'A   Chromosome distribution', '#0072B2'),
                 (gene_counts.value_counts().sort_index(), 'B   Unique gene annotations per CpG', '#D55E00'),
                 (regions, 'C   Gene-region annotations', '#009E73'),
                 (islands, 'D   CpG-island context', '#CC79A7')]
        for ax, (counts, title, color) in zip(axes.flat, specs):
            y = np.arange(len(counts))
            ax.barh(y, counts.values, color=color, height=.65)
            labels = [str(v).replace('_', ' ') for v in counts.index]
            ax.set_yticks(y, labels); ax.invert_yaxis()
            for i, count in enumerate(counts):
                ax.text(count + .025 * max(counts), i, f'{count} ({count/n:.1%})', va='center', fontsize=9)
            ax.set_xlim(0, max(counts) * 1.48)
            ax.set_title(title, loc='left', fontweight='bold', pad=12)
            ax.set_xlabel('Number of CpG sites')
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
            ax.spines[['top', 'right', 'left']].set_visible(False)
            ax.tick_params(axis='y', length=0); ax.set_axisbelow(True)
            ax.grid(axis='x', color='#e7e7e7', linewidth=.6)
        fig.suptitle(f'{disease} | Genomic context of {n} selected CpG sites', fontsize=15, fontweight='bold', y=.98)
        fig.text(.04, .025, 'Labels: count (% of selected CpGs). Gene-region categories can overlap.\n'
                 'GPL13534 legacy annotations; blank island context is grouped as unannotated / open sea.', fontsize=9)
        fig.subplots_adjust(left=.19, right=.97, bottom=.14, top=.88, hspace=.43, wspace=.95)
    return fig


def reactome_figure(result, n, background, disease):
    top = result.loc[result.overlap_CpGs.gt(0)].sort_values(['FDR', 'p_value', 'pathway']).head(12).copy()
    significant = int(result.FDR.lt(.05).sum())
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(12, 8.5))
        y = np.arange(len(top))
        score = -np.log10(top.p_value.clip(lower=1e-300))
        for i, x in enumerate(score):
            ax.plot([0, x], [i, i], color='#d4d4d4', lw=1.5, zorder=1)
        ax.scatter(score, y, s=35 + 30*top.overlap_CpGs, color='#0072B2', edgecolor='white', linewidth=.7, zorder=3)
        ax.set_yticks(y, [textwrap.fill(v, 43) for v in top.pathway], fontsize=9)
        ax.invert_yaxis()
        limit = max(float(score.max()) if len(score) else 1, 1)
        ax.set_xlim(0, limit*1.65)
        for i, row in enumerate(top.itertuples()):
            ax.text(limit*1.12, i, f'{row.overlap_CpGs}/{n}   |   {row.FDR:.3g}', va='center', fontsize=9)
        ax.text(limit*1.12, 1.03, 'CpGs   |   BH FDR', transform=ax.get_xaxis_transform(), fontsize=9, fontweight='bold')
        ax.set_xlabel('−log₁₀(nominal p-value)')
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(axis='y', length=0, pad=10)
        ax.set_axisbelow(True); ax.grid(axis='x', color='#e7e7e7', linewidth=.6)
        fig.suptitle(f'{disease} | Reactome pathway associations', fontsize=15, fontweight='bold', y=.98)
        fig.text(.04, .92, f'{significant} of {len(result):,} pathways significant at BH FDR < 0.05', fontsize=12,
                 fontweight='bold', color='#444444')
        fig.text(.04, .025, f'Top {len(top)} pathways with CpG overlap, ranked by BH FDR then nominal p-value. Dot area increases with overlap.\n'
                 f'One-sided exact CpG-membership test; background: {background:,} CpGs. BH correction includes all tested pathways.\n'
                 'Nominal associations are exploratory; they do not demonstrate pathway activation or repression.', fontsize=9)
        fig.subplots_adjust(left=.40, right=.98, top=.83, bottom=.16)
    return fig


def render_saved(out, disease):
    out = Path(out)
    selected = pd.read_csv(out/'data/candidate_genomic_annotation.tsv', sep='\t', dtype=str)
    result = pd.read_csv(out/'data/reactome_CpG_set_enrichment.tsv', sep='\t')
    status = json.loads((out/'data/annotation_analysis_status.json').read_text())
    save_publication(annotation_figure(selected, disease), out, 'genomic_annotation')
    save_publication(reactome_figure(result, len(selected), status['pathways']['background_CpGs'], disease), out, 'reactome_enrichment')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--disease', required=True)
    args = parser.parse_args()
    render_saved(args.out, args.disease)
