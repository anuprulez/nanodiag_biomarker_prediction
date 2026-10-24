#!/usr/bin/env python3
"""Select top 100 hyper/hypomethylated CpGs using healthy stability filters."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
PAPER = 'https://www.nature.com/articles/s41375-021-01316-z'
DELTA = 'deltaBeta_AML_minus_Normal'


def select_ranked(stats, top_n=100):
    eligible = stats.loc[((stats['mean_Normal'] < 0.1) | (stats['mean_Normal'] > 0.9)) &
                         (stats['sd_Normal'] < 0.05)].copy()
    hyper = eligible.loc[eligible[DELTA] > 0].sort_values([DELTA, 'CpG_ID'], ascending=[False, True]).head(top_n).copy()
    hypo = eligible.loc[eligible[DELTA] < 0].sort_values([DELTA, 'CpG_ID'], ascending=[True, True]).head(top_n).copy()
    if len(hyper) != top_n or len(hypo) != top_n:
        raise ValueError(f'Not enough eligible sites: hyper={len(hyper)}, hypo={len(hypo)}')
    for direction, frame in [('hyper', hyper), ('hypo', hypo)]:
        frame['direction'] = direction
        frame['rank_within_direction'] = range(1, len(frame) + 1)
    return eligible, hyper, hypo


def run(args):
    metadata = pd.read_csv(args.metadata, sep='\t', dtype=str)
    if not metadata['sample_id'].is_unique:
        raise ValueError('Duplicate sample IDs in metadata')
    controls = metadata.loc[metadata['group'].eq('healthy_control'), 'sample_id'].tolist()
    aml = metadata.loc[metadata['group'].eq('AML'), 'sample_id'].tolist()
    if len(controls) != 464 or len(aml) != 62:
        raise ValueError('Expected 464 controls and 62 AML samples')
    if not metadata.loc[metadata['sample_id'].isin(controls), 'series'].eq('GSE50660').all():
        raise ValueError('Controls must come from GSE50660')
    if not metadata.loc[metadata['sample_id'].isin(aml), 'series'].eq('GSE58477').all():
        raise ValueError('AML samples must come from GSE58477')
    stats_parts = []
    for chunk in pd.read_csv(args.matrix, sep='\t', chunksize=10000):
        if chunk.columns[0] != 'ID_REF' or chunk.columns[1:].tolist() != metadata['sample_id'].tolist():
            raise ValueError('Matrix sample columns disagree with metadata')
        probes = chunk['ID_REF'].astype(str)
        if not probes.str.fullmatch(r'cg\d+').all():
            raise ValueError('Use the original unexpanded CpG matrix')
        normal = chunk[controls].to_numpy(dtype=np.float64)
        disease = chunk[aml].to_numpy(dtype=np.float64)
        for values in (normal, disease):
            if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
                raise ValueError('Expected complete finite beta values in [0,1]')
        mean_normal = normal.mean(axis=1)
        mean_aml = disease.mean(axis=1)
        stats_parts.append(pd.DataFrame({'CpG_ID': probes.to_numpy(), 'mean_Normal': mean_normal,
                                        'sd_Normal': normal.std(axis=1, ddof=args.sd_ddof),
                                        'mean_AML': mean_aml, DELTA: mean_aml - mean_normal}))
    stats = pd.concat(stats_parts, ignore_index=True)
    if not stats['CpG_ID'].is_unique:
        raise ValueError('Duplicate CpG IDs')
    eligible, hyper, hypo = select_ranked(stats, args.top_n)
    annotation = pd.read_csv(args.annotation, sep='\t', dtype=str, keep_default_na=False).set_index('ID')
    if not annotation.index.is_unique:
        raise ValueError('Duplicate annotation IDs')
    selected = pd.concat([hyper, hypo], ignore_index=True)
    if not selected['CpG_ID'].is_unique:
        raise ValueError('Hyper and hypo lists overlap')
    selected['Gene'] = selected['CpG_ID'].map(annotation['UCSC_RefGene_Name']).fillna('').map(
        lambda value: ';'.join(dict.fromkeys(g.strip() for g in value.split(';') if g.strip())))
    selected['CpG_Gene'] = selected['CpG_ID'] + '_' + selected['Gene']
    columns = ['CpG_ID', 'Gene', DELTA, 'CpG_Gene']
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected[columns].to_csv(args.output_dir / 'aml_top200_cpg_gene_delta_beta.tsv', sep='\t', index=False, float_format='%.15g')
    for direction in ['hyper', 'hypo']:
        selected.loc[selected['direction'].eq(direction), columns].to_csv(
            args.output_dir / f'aml_top100_{direction}_cpg_gene_delta_beta.tsv', sep='\t', index=False, float_format='%.15g')
    selected.to_csv(args.output_dir / 'aml_top200_cpg_statistics.tsv', sep='\t', index=False, float_format='%.15g')
    eligible.to_csv(args.output_dir / 'aml_eligible_cpg_statistics.tsv.gz', sep='\t', index=False, float_format='%.15g')
    report = dict(paper=PAPER, matrix=str(args.matrix.resolve()), annotation=str(args.annotation.resolve()),
                  normal_samples=len(controls), aml_samples=len(aml), input_cpgs=len(stats),
                  passed_control_mean=int(((stats['mean_Normal'] < .1) | (stats['mean_Normal'] > .9)).sum()),
                  passed_mean_and_sd=len(eligible), eligible_hyper=int((eligible[DELTA] > 0).sum()),
                  eligible_hypo=int((eligible[DELTA] < 0).sum()), selected_hyper=len(hyper), selected_hypo=len(hypo),
                  selected_without_gene=int(selected['Gene'].eq('').sum()),
                  selected_multiple_genes=int(selected['Gene'].str.contains(';', regex=False).sum()),
                  sd_ddof=args.sd_ddof, delta_definition='mean_AML - mean_Normal',
                  hyper_delta_range=[float(hyper[DELTA].min()), float(hyper[DELTA].max())],
                  hypo_delta_range=[float(hypo[DELTA].min()), float(hypo[DELTA].max())],
                  gene_policy='annotate after ranking; retain blanks and semicolon-separated multiple genes',
                  extra_probe_filters='none; all complete CpGs in original harmonised matrix',
                  missing_policy='input harmonisation excluded CpGs missing in any selected sample',
                  tie_break='CpG ID ascending')
    (args.output_dir / 'aml_top200_selection_report.json').write_text(json.dumps(report, indent=2) + '\n')
    # Verify the files actually written, including thresholds, signs and row order.
    saved = pd.read_csv(args.output_dir / 'aml_top200_cpg_statistics.tsv', sep='\t')
    assert len(saved) == args.top_n * 2 and saved['CpG_ID'].is_unique
    assert ((saved['mean_Normal'] < .1) | (saved['mean_Normal'] > .9)).all()
    assert (saved['sd_Normal'] < .05).all()
    assert np.allclose(saved[DELTA], saved['mean_AML'] - saved['mean_Normal'], atol=1e-13, rtol=0)
    for direction, expected in [('hyper', hyper), ('hypo', hypo)]:
        actual = saved.loc[saved['direction'].eq(direction)]
        assert actual['CpG_ID'].tolist() == expected['CpG_ID'].tolist()
        assert (actual[DELTA] > 0).all() if direction == 'hyper' else (actual[DELTA] < 0).all()
    print(json.dumps(report, indent=2))
    print('Top hyper:\n', selected.loc[selected['direction'].eq('hyper'), ['CpG_ID', 'Gene', DELTA]].head().to_string(index=False))
    print('Top hypo:\n', selected.loc[selected['direction'].eq('hypo'), ['CpG_ID', 'Gene', DELTA]].head().to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', type=Path, default=HERE / 'raw/harmonised/combined_beta.tsv.gz')
    parser.add_argument('--metadata', type=Path, default=HERE / 'raw/harmonised/sample_metadata.tsv')
    parser.add_argument('--annotation', type=Path, default=HERE / 'raw/GPL13534_probe_annotation.tsv')
    parser.add_argument('--output-dir', type=Path, default=HERE / 'raw')
    parser.add_argument('--top-n', type=int, default=100)
    parser.add_argument('--sd-ddof', type=int, choices=[0, 1], default=1)
    run(parser.parse_args())
