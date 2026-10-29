#!/usr/bin/env python3
"""Plot methylation-profile correlations with neighbors shown in a local XAI PDF.

Example (from aml_analysis):
    python train_gnn/plot_xai_neighbor_correlations.py --node-id 7980

The displayed PDF legend determines the neighborhood, not a fresh graph layout
or a new explanation. Green means seed score > 0, matching draw_xai_local_graph.
Correlations use sample methylation values, excluding NeDBIT features and scores.
"""
import argparse
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT/'data_corr_0.33_knn_n_3000_n_neg_10000'


def plotted_nodes(pdf):
    import pymupdf
    with pymupdf.open(pdf) as document:
        text = '\n'.join(page.get_text() for page in document)
    matches = re.findall(r'\b(\d+)\s*:\s*(cg\d+_[^\s]+)', text)
    nodes = {}
    for node, name in matches:
        node = int(node)
        if node in nodes and nodes[node] != name:
            raise ValueError(f'Conflicting labels for node {node} in {pdf}')
        nodes[node] = name
    if not nodes:
        raise ValueError(f'No node-ID:CpG_gene legend entries found in {pdf}')
    return nodes


def read_profiles(path, names):
    selected = []
    for chunk in pd.read_csv(path, index_col=0, chunksize=512):
        selected.append(chunk.loc[chunk.index.isin(names)])
    profiles = pd.concat(selected)
    if profiles.index.duplicated().any():
        raise ValueError('Methylation profiles must have unique CpG-gene names')
    missing = set(names)-set(profiles.index)
    if missing:
        raise ValueError(f'Missing methylation profiles: {sorted(missing)}')
    return profiles.loc[list(names)].apply(pd.to_numeric, errors='raise')


def correlate_profiles(profiles, nodes, table, center, method='pearson'):
    if center not in nodes:
        raise ValueError(f'XAI node {center} is absent from the plotted neighborhood')
    if table.index.duplicated().any():
        raise ValueError('Node IDs must be unique')
    for node, name in nodes.items():
        if node not in table.index or table.loc[node, 'name'] != name:
            raise ValueError(f'Node table and XAI PDF disagree for {node}:{name}')
    x = profiles.loc[nodes[center]].to_numpy(float)
    rows = []
    for node, name in nodes.items():
        if node == center:
            continue
        y = profiles.loc[name].to_numpy(float)
        valid = np.isfinite(x) & np.isfinite(y)
        count = int(valid.sum())
        status = 'ok'
        if count < 3:
            r, status = np.nan, 'fewer_than_three_paired_samples'
        elif np.ptp(x[valid]) == 0 or np.ptp(y[valid]) == 0:
            r, status = np.nan, 'constant_profile'
        else:
            statistic = pearsonr if method == 'pearson' else spearmanr
            r = float(statistic(x[valid], y[valid]).statistic)
        seed_score = float(table.loc[node, 'seed_score'])
        if not np.isfinite(seed_score):
            raise ValueError(f'Nonfinite seed score for {node}')
        rows.append(dict(xai_node_id=center, xai_node_name=nodes[center],
                         neighbor_node_id=node, neighbor_name=name,
                         node_group='green_seed' if seed_score > 0 else 'blue_other',
                         seed_score=seed_score, method=method, correlation=r,
                         n_paired_samples=count, status=status))
    if not rows:
        raise ValueError('No neighboring nodes in the selected plot')
    return pd.DataFrame(rows).sort_values(['node_group', 'correlation'],
                                         ascending=[False, False], na_position='last')


def correlation_plot(results, center, center_name, method):
    fig, ax = plt.subplots(figsize=(14, max(8, .65*len(results)+2.5)), constrained_layout=True)
    positions = np.arange(len(results))
    colors = results.node_group.map({'green_seed': 'green', 'blue_other': 'tab:blue'})
    ax.barh(positions, results.correlation, color=colors, height=.7)
    ax.set_yticks(positions, [f'{row.neighbor_node_id}: {row.neighbor_name}'
                             for row in results.itertuples()])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.set_xticks(np.linspace(0, 1, 6))
    symbol = 'r' if method == 'pearson' else 'ρ'
    ax.set_xlabel(f'{method.capitalize()} correlation ({symbol}) across methylation samples', fontsize=28.8)
    ax.set_title(f'AML: methylation correlation with XAI node {center}\n{center_name}', fontsize=21, pad=18)
    for position, row in enumerate(results.itertuples()):
        if np.isfinite(row.correlation):
            value = row.correlation
            ax.text(value+(.025 if value >= 0 else -.025), position, f'{value:.3f}',
                    va='center', ha='left' if value >= 0 else 'right', fontsize=21.6)
        else:
            ax.text(.02, position, 'N/A', va='center', fontsize=21.6)
    ax.axvline(0, color='#555555', linewidth=1)
    ax.set_axisbelow(True)
    ax.grid(axis='x', color='#dddddd', linewidth=.7)
    ax.tick_params(axis='both', labelsize=21.6)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.legend(handles=[Patch(color='green', label='Green: seed nodes'),
                       Patch(color='tab:blue', label='Blue: other likely positive predicted nodes')],
              loc='upper center', bbox_to_anchor=(.5, -.14), ncol=2,
              frameon=False, fontsize=24)
    return fig


def correlate_all_seeds(profiles, table, center, method='pearson'):
    """Compare the RN with every positive-weight seed, not just local neighbors."""
    seeds = table.loc[table.seed_score.gt(0)]
    nodes = {int(node): row['name'] for node, row in seeds.iterrows() if int(node) != center}
    if not nodes:
        raise ValueError('No seed nodes available for RN correlations')
    nodes[center] = table.loc[center, 'name']
    return correlate_profiles(profiles, nodes, table, center, method)


def all_seed_correlation_plot(results, center, center_name, method):
    ordered = results.sort_values('correlation', na_position='last').reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(14, 10), constrained_layout=True)
    ax.bar(np.arange(1, len(ordered)+1), ordered.correlation, color='green', width=.85,
           label=f'All seed nodes (n={len(ordered)})')
    ax.axhline(0, color='#555555', linewidth=1)
    ax.set_xlim(.25, len(ordered)+.75)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel('Seed nodes ordered by correlation with RN', fontsize=28.8)
    ax.set_ylabel(f'{method.capitalize()} correlation with RN', fontsize=28.8)
    ax.set_title(f'RN {center}: correlation with all seed nodes\n{center_name}', fontsize=21, pad=18)
    ax.tick_params(labelsize=21.6)
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#dddddd', linewidth=.7)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=21.6, loc='upper left', frameon=False)
    return fig


def delta_beta_statistics(profiles, metadata_path, matrix_path):
    """Check positional sample alignment before comparing AML and controls."""
    metadata = pd.read_csv(metadata_path, sep='\t')
    samples = pd.read_csv(matrix_path, sep='\t', nrows=0).columns[1:].tolist()
    if metadata.sample_id.duplicated().any() or metadata.sample_id.tolist() != samples:
        raise ValueError('Metadata sample order differs from original annotated matrix')
    if len(metadata) != profiles.shape[1]:
        raise ValueError('Methylation sample count differs from metadata')
    columns = list(map(str, profiles.columns))
    if columns != samples and columns != [str(i) for i in range(len(samples))]:
        raise ValueError('Methylation profile sample columns have an unexpected order')
    values = profiles.to_numpy(float)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError('Delta beta requires finite methylation beta values between 0 and 1')
    healthy = metadata.group.eq('healthy_control').to_numpy()
    aml = metadata.group.eq('AML').to_numpy()
    if not healthy.any() or not aml.any() or not np.all(healthy | aml):
        raise ValueError('Expected healthy_control and AML sample groups')
    stats = pd.DataFrame(index=profiles.index)
    stats['healthy_mean_beta'] = values[:, healthy].mean(axis=1)
    stats['AML_mean_beta'] = values[:, aml].mean(axis=1)
    stats['delta_beta_AML_minus_healthy'] = stats.AML_mean_beta-stats.healthy_mean_beta
    stats['healthy_n'], stats['AML_n'] = int(healthy.sum()), int(aml.sum())
    return stats


def cohort_comparison_table(stats, nodes, table, center, external_path):
    """Match by unique CpG using the same external effect source as compare-predictions."""
    columns = ['CpG_ID', 'Healthy_GSE40279_mean_beta', 'AML_GSE62298_mean_beta',
               'Delta_beta_AML_minus_healthy']
    probes = {name.split('_', 1)[0] for name in nodes.values()}
    selected = [chunk.loc[chunk.CpG_ID.isin(probes), columns]
                for chunk in pd.read_csv(external_path, sep='\t', usecols=columns, chunksize=10000)]
    external = pd.concat(selected, ignore_index=True)
    if external.groupby('CpG_ID')[columns[1:]].nunique(dropna=False).gt(1).any().any():
        raise ValueError('External gene mappings contain inconsistent CpG statistics')
    external = external.drop_duplicates('CpG_ID').rename(columns={
        'Delta_beta_AML_minus_healthy': 'external_delta_beta_AML_minus_healthy'})
    if not np.allclose(external.AML_GSE62298_mean_beta-external.Healthy_GSE40279_mean_beta,
                       external.external_delta_beta_AML_minus_healthy, equal_nan=True):
        raise ValueError('External delta beta does not agree with AML and control means')
    records = pd.DataFrame([dict(node_id=node, node_name=name, CpG_ID=name.split('_',1)[0],
                                 node_group='xai_node' if node == center else
                                 ('green_seed' if table.loc[node, 'seed_score'] > 0 else 'blue_other'))
                            for node, name in nodes.items()])
    records = records.join(stats, on='node_name', validate='many_to_one')
    records = records.merge(external, on='CpG_ID', how='left', validate='many_to_one', indicator=True)
    records['matched_external'] = records.pop('_merge').eq('both')
    return records


def delta_beta_scatter(comparison, center, center_name):
    xfield, yfield = 'delta_beta_AML_minus_healthy', 'external_delta_beta_AML_minus_healthy'
    valid = comparison.loc[np.isfinite(comparison[[xfield, yfield]]).all(axis=1)].copy()
    if valid.empty:
        raise ValueError('No displayed XAI nodes have finite current and external delta beta')
    fig, ax = plt.subplots(figsize=(14, 11), constrained_layout=True)
    for group, color, label, marker in [('green_seed', 'green', 'Green: seed nodes', 'o'),
                                       ('blue_other', 'tab:blue', 'Blue: other plotted nodes', 'o'),
                                       ('xai_node', '#c43b3b', f'XAI node {center}', '*')]:
        subset = valid.loc[valid.node_group.eq(group)]
        ax.scatter(subset[xfield], subset[yfield], color=color, label=label,
                   marker=marker, s=320 if group == 'xai_node' else 140,
                   edgecolors='white', linewidth=.7, zorder=4 if group == 'xai_node' else 3)
    low = min(-.08, float(valid[[xfield, yfield]].min().min())-.1)
    high = max(.3, float(valid[[xfield, yfield]].max().max())+.2)
    ax.plot([low, high], [low, high], '--', color='#777777', linewidth=1.1,
            label='Equal effect (y = x)', zorder=1)
    ax.axhline(0, color='#bbbbbb', linewidth=.8)
    ax.axvline(0, color='#bbbbbb', linewidth=.8)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    from matplotlib.ticker import MaxNLocator
    ticks = MaxNLocator(nbins=6).tick_values(low, high)
    ticks = ticks[(ticks >= low) & (ticks <= high)]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Current cohort Δβ (AML − normal controls)', fontsize=28.8)
    ax.set_ylabel('External cohort Δβ (AML − normal controls)', fontsize=28.8)
    ax.set_title(f'AML: current versus external cohort methylation effects\n'
                 f'XAI node {center}: {center_name} and its displayed neighbors', fontsize=20, pad=16)
    ax.tick_params(labelsize=21.6)
    ax.set_axisbelow(True)
    ax.grid(color='#dddddd', linewidth=.6)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.legend(loc='upper left', fontsize=21.6, frameon=True, facecolor='white')
    place_cpg_labels(fig, ax, valid, xfield, yfield)
    return fig


def place_cpg_labels(fig, ax, data, xfield, yfield):
    """Place full CpG_Gene labels near their points without leader lines."""
    from matplotlib.transforms import Bbox
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    occupied = [ax.get_legend().get_window_extent(renderer)]
    point_boxes = [Bbox.from_bounds(x-7, y-7, 14, 14)
                   for x, y in ax.transData.transform(data[[xfield, yfield]].to_numpy())]
    bounds = ax.get_window_extent(renderer)
    for row in data.sort_values(yfield, ascending=False).itertuples():
        annotation = ax.annotate(row.node_name, (getattr(row, xfield), getattr(row, yfield)),
                                 xytext=(8, 8), textcoords='offset points', fontsize=19.2,
                                 color='#b52a2a' if row.node_group == 'xai_node' else '#222222')
        candidates = [(dx, dy, align) for distance in (10, 26, 44, 66, 92, 122, 156)
                      for dx, dy, align in ((distance, distance, 'left'),
                                            (-distance, distance, 'right'),
                                            (distance, -distance, 'left'),
                                            (-distance, -distance, 'right'),
                                            (distance, 0, 'left'),
                                            (-distance, 0, 'right'))]
        best, best_score = None, float('inf')
        for dx, dy, align in candidates:
            annotation.set_position((dx, dy))
            annotation.set_ha(align)
            box = annotation.get_window_extent(renderer).expanded(1.03, 1.15)
            inside = bounds.contains(box.x0, box.y0) and bounds.contains(box.x1, box.y1)
            score = sum(box.overlaps(other) for other in occupied+point_boxes)+100*(not inside)
            if score < best_score:
                best, best_score = (dx, dy, align, box), score
            if score == 0:
                break
        dx, dy, align, box = best
        annotation.set_position((dx, dy))
        annotation.set_ha(align)
        occupied.append(box)


def combined_xai_figure(xai_pdf, bar_pdf, delta_pdf, output, center, center_name, dpi=300, prediction_group='LP'):
    """Compose two RN panels or three LP panels, preserving vector content."""
    import pymupdf
    output = Path(output)
    with pymupdf.open() as document:
        page = document.new_page(width=1440, height=780 if prediction_group == 'RN' else 1401.4)
        page.insert_textbox(pymupdf.Rect(24, 12, 1416, 66),
                            f'AML: XAI node {center} ({center_name})',
                            fontsize=28, fontname='helv', align=1)
        from matplotlib.font_manager import FontProperties, findfont
        page.insert_font(fontname='PanelTitle', fontfile=str(findfont(FontProperties(weight='bold'))))
        panels = [('A. Explanation subgraph', xai_pdf, (24, 83, 1416, 736.4)),
                  ('B. All seeds: correlation with RN' if prediction_group == 'RN' else 'B. DNA methylation correlation', bar_pdf, (24, 771.4, 710, 1371.4)),
                  ('C. Δβ : Original vs external cohort', delta_pdf, (734, 771.4, 1416, 1371.4))]
        if prediction_group == 'RN':
            panels = [('A. Explanation subgraph', xai_pdf, (24, 90, 710, 750)),
                      ('B. All seeds: correlation with RN', bar_pdf, (734, 90, 1416, 750))]
        for title, source_path, coords in panels:
            rect = pymupdf.Rect(coords)
            page.insert_textbox(pymupdf.Rect(rect.x0, rect.y0, rect.x1, rect.y0+45),
                                title, fontsize=24, fontname='PanelTitle', align=0)
            with pymupdf.open(source_path) as source:
                if len(source) != 1:
                    raise ValueError(f'{source_path}: expected one plot page')
                # Use the panel heading in the composite; omit standalone plot titles.
                clip = pymupdf.Rect(source[0].rect)
                title_bottoms = [span['bbox'][3]
                    for block in source[0].get_text('dict')['blocks']
                    for line in block.get('lines', []) for span in line['spans']
                    if span['bbox'][1] < source[0].rect.height*.2 and
                    (span['text'].startswith(('AML:', 'Breast Cancer:',
                                              'Explanation subgraph of ', 'RN '))
                     or (span['text'].startswith('XAI node ') and ':' in span['text'])
                     or span['text'] == center_name)]
                if title_bottoms:
                    clip.y0 = max(title_bottoms)+6
                page.show_pdf_page(pymupdf.Rect(rect.x0, rect.y0+45, rect.x1, rect.y1),
                                   source, 0, keep_proportion=True, clip=clip)
        document.save(output.with_suffix('.pdf'), garbage=4, deflate=True)
        page.get_pixmap(dpi=max(300, int(dpi)), alpha=False).save(output.with_suffix('.png'))
    print(output.with_suffix('.pdf'))
    print(output.with_suffix('.png'))


def create_neighbor_correlation_plots(dataset, node_id, model='PNA', *, xai_plot=None,
                                      features=None, node_table=None, metadata=None,
                                      matrix=None, external=None, output=None, method='pearson', dpi=300,
                                      data_dir=None, plot_dir=None, prediction_group=None):
    """Shared standalone/stage entry point; consumes the exact current XAI PDF."""
    dataset = Path(dataset).resolve()
    generated = dataset/'generated'
    if xai_plot is None:
        candidates = list((generated/'plots').glob(
            f'Explanation_local_subgraph_node_{node_id}_edges_*_epo_*_{model}.pdf'))
        if len(candidates) != 1:
            raise ValueError(f'Expected one local XAI PDF; found {len(candidates)}. Supply --xai-plot.')
        xai_plot = candidates[0]
    nodes = plotted_nodes(xai_plot)
    if prediction_group is None:
        import pymupdf
        with pymupdf.open(xai_plot) as source:
            prediction_group = 'RN' if '(RN)' in source[0].get_text() else 'LP'
    prediction_group = prediction_group.upper()
    if prediction_group not in ('LP', 'RN'):
        raise ValueError('Prediction group must be LP or RN')
    table_path = Path(node_table or dataset/'preprocessed/out_genes_aml.csv')
    table = pd.read_csv(table_path, sep=r'\s+', header=None,
                        names=['node_id', 'name', 'seed_score']).set_index('node_id')
    features = Path(features or generated/'data'/f'dnam_features_aml_{model}.csv')
    metadata = Path(metadata or ROOT/'raw/harmonised/sample_metadata.tsv')
    matrix = Path(matrix or ROOT/'raw/harmonised/combined_beta_cpg_gene.tsv.gz')
    external = Path(external or ROOT/'evaluation/results/cpg_gene_beta_comparison.tsv')
    requested_names = list(nodes.values())
    if prediction_group == 'RN':
        requested_names = list(dict.fromkeys(requested_names+table.loc[table.seed_score.gt(0), 'name'].tolist()))
    profiles = read_profiles(features, requested_names)
    results = (correlate_all_seeds(profiles, table, node_id, method) if prediction_group == 'RN'
               else correlate_profiles(profiles, nodes, table, node_id, method))
    if prediction_group == 'RN':
        # RN figures require methylation profiles only, not external comparisons.
        output = Path(output or generated)
        data_dir = Path(data_dir or output/'data')
        plot_dir = Path(plot_dir or output/'plots')
        for folder in (data_dir, plot_dir):
            folder.mkdir(parents=True, exist_ok=True)
        seed_stem = f'XAI_all_seed_correlations_node_{node_id}_{model}_{method}'
        results.to_csv(data_dir/f'{seed_stem}.tsv', sep='\t', index=False)
        fig = all_seed_correlation_plot(results, node_id, nodes[node_id], method)
        with matplotlib.rc_context({'pdf.fonttype': 42, 'svg.fonttype': 'none'}):
            for extension in ('pdf', 'png', 'svg'):
                fig.savefig(plot_dir/f'{seed_stem}.{extension}', dpi=max(300, dpi),
                            bbox_inches='tight', facecolor='white')
        plt.close(fig)
        combined_xai_figure(xai_plot, plot_dir/f'{seed_stem}.pdf', None,
                            plot_dir/f'XAI_combined_node_{node_id}_{model}',
                            node_id, nodes[node_id], dpi, prediction_group='RN')
        provenance = dict(prediction_group='RN', model=model, node_id=node_id,
                          node_name=nodes[node_id], n_seeds=len(results),
                          seed_scope='All graph nodes with seed_score > 0, excluding self',
                          sample_count=profiles.shape[1], method=method,
                          xai_plot=str(Path(xai_plot).resolve()),
                          features=str(features.resolve()), node_table=str(table_path.resolve()),
                          panels=['Explanation subgraph', 'All-seed correlation with RN'])
        (data_dir/f'{seed_stem}.json').write_text(json.dumps(provenance, indent=2)+'\n')
        print(f'RN {node_id}: included all {len(results)} seed nodes; two-panel composite generated.')
        return results
    stats = delta_beta_statistics(profiles, metadata, matrix)
    results = results.join(stats, on='neighbor_name', validate='many_to_one')
    comparison_nodes = dict(nodes)
    if prediction_group == 'RN':
        comparison_nodes.update({int(node): row['name']
                                 for node, row in table.loc[table.seed_score.gt(0)].iterrows()})
    all_comparison = cohort_comparison_table(stats, comparison_nodes, table, node_id, external)
    comparison = all_comparison.loc[all_comparison.node_id.isin(nodes)].copy()
    external_by_name = all_comparison.set_index('node_name')[['external_delta_beta_AML_minus_healthy', 'matched_external']]
    results = results.join(external_by_name, on='neighbor_name', validate='many_to_one')
    center_delta = float(stats.loc[nodes[node_id], 'delta_beta_AML_minus_healthy'])
    results['xai_delta_beta_AML_minus_healthy'] = center_delta
    output = Path(output or generated)
    data_dir, plot_dir = Path(data_dir or output/'data'), Path(plot_dir or output/'plots')
    for folder in (data_dir, plot_dir):
        folder.mkdir(parents=True, exist_ok=True)
    stem = f'XAI_neighbor_correlations_node_{node_id}_{model}_{method}'
    results.to_csv(data_dir/f'{stem}.tsv', sep='\t', index=False)
    seed_stem = f'XAI_all_seed_correlations_node_{node_id}_{model}_{method}'
    if prediction_group == 'RN':
        results.to_csv(data_dir/f'{seed_stem}.tsv', sep='\t', index=False)
    comparison.to_csv(data_dir/f'XAI_neighbor_delta_beta_current_vs_external_node_{node_id}_{model}.tsv', sep='\t', index=False)
    missing = comparison.loc[~comparison.matched_external, 'node_name'].tolist()
    if missing:
        print(f'Nodes absent from external cohort (excluded from scatter): {missing}', flush=True)
    for fig, name in (
            ((all_seed_correlation_plot(results, node_id, nodes[node_id], method) if prediction_group == 'RN'
              else correlation_plot(results, node_id, nodes[node_id], method)), seed_stem if prediction_group == 'RN' else stem),
            (delta_beta_scatter(comparison, node_id, nodes[node_id]),
             f'XAI_neighbor_delta_beta_current_vs_external_node_{node_id}_{model}')):
        with matplotlib.rc_context({'pdf.fonttype': 42, 'svg.fonttype': 'none'}):
            for extension in ('pdf', 'png', 'svg'):
                path = plot_dir/f'{name}.{extension}'
                fig.savefig(path, dpi=max(300, dpi), bbox_inches='tight', facecolor='white')
                print(path)
        plt.close(fig)
    combined_xai_figure(
        xai_plot, plot_dir/f"{seed_stem if prediction_group == 'RN' else stem}.pdf",
        plot_dir/f'XAI_neighbor_delta_beta_current_vs_external_node_{node_id}_{model}.pdf',
        plot_dir/f'XAI_combined_node_{node_id}_{model}', node_id, nodes[node_id], dpi, prediction_group)
    provenance = dict(xai_plot=str(Path(xai_plot).resolve()), features=str(features.resolve()),
                      node_table=str(table_path.resolve()), metadata=str(metadata.resolve()),
                      matrix=str(matrix.resolve()), external=str(external.resolve()), method=method,
                      prediction_group=prediction_group,
                      seed_scope='All graph nodes with seed_score > 0' if prediction_group == 'RN' else 'Displayed neighbors',
                      n_seed_correlations=len(results) if prediction_group == 'RN' else None,
                      neighborhood='All displayed nodes in cohort scatter; all seeds in RN correlation panel',
                      sample_count=profiles.shape[1],
                      correlation='Pairwise finite sample methylation values; no NeDBIT or seed-score features',
                      delta_beta='Current and external cohort mean AML beta minus mean healthy-control beta; scatter x=current, y=external',
                      center_node_id=node_id, center_name=nodes[node_id],
                      center_delta_beta=center_delta)
    (data_dir/f'{stem}.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(results[['neighbor_node_id', 'neighbor_name', 'node_group', 'correlation',
                   'delta_beta_AML_minus_healthy', 'n_paired_samples']].to_string(index=False))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--node-id', type=int, default=7980)
    parser.add_argument('--model', default='PNA')
    parser.add_argument('--xai-group', dest='prediction_group', type=str.upper, choices=('LP','RN'),
                        help='Default: infer LP/RN from the source XAI plot title')
    parser.add_argument('--xai-plot', type=Path)
    parser.add_argument('--features', type=Path)
    parser.add_argument('--node-table', type=Path)
    parser.add_argument('--metadata', type=Path)
    parser.add_argument('--external', type=Path, help='External CpG cohort comparison TSV used by compare-predictions')
    parser.add_argument('--matrix', type=Path, help='Original annotated beta matrix; verifies sample order')
    parser.add_argument('--method', choices=('pearson', 'spearman'), default='pearson')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()
    create_neighbor_correlation_plots(**vars(args))


if __name__ == '__main__':
    main()
