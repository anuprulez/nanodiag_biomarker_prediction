"""Export exhaustive frozen candidate CpG tables from saved post-evaluation inputs."""
from pathlib import Path
import argparse
import json
import hashlib
import numpy as np
import pandas as pd


def run(out, disease):
    out = Path(out)
    expected = {'AML': 17, 'BC': 33}[disease]
    paths = [out / 'data/fixed_candidate_panel.tsv',
             out / 'data/candidate_genomic_annotation.tsv']
    panel, annotation = [pd.read_csv(p, sep='\t') for p in paths]
    for name, frame in [('panel', panel), ('annotation', annotation)]:
        if len(frame) != expected or frame.CpG_ID.isna().any() or not frame.CpG_ID.is_unique:
            raise ValueError(f'{disease}: {name} must contain {expected} unique CpGs')
    if set(panel.CpG_ID) != set(annotation.CpG_ID):
        raise ValueError('Panel and annotation CpG identities differ')
    frame = panel.merge(annotation, on='CpG_ID', validate='one_to_one', sort=False)
    external = f'Delta_beta_{disease}_minus_healthy'
    original = f'Internal_delta_beta_{disease}_minus_healthy'
    for column in [external, original]:
        values = pd.to_numeric(frame[column], errors='raise').to_numpy(float)
        if not np.isfinite(values).all() or (np.abs(values) > 1).any():
            raise ValueError(f'Invalid or missing delta beta: {column}')
    def labels(value, missing):
        if pd.isna(value) or not str(value).strip():
            return missing
        return ';'.join(dict.fromkeys(x.strip() for x in str(value).split(';') if x.strip()))
    table = pd.DataFrame({
        'CpG_site': frame.CpG_ID,
        'Gene': frame.UCSC_RefGene_Name.map(lambda x: labels(x, 'Unannotated')),
        'Delta_beta_external': frame[external],
        'Delta_beta_original': frame[original],
        'Gene_locations': frame.UCSC_RefGene_Group.map(lambda x: labels(x, 'Unannotated')),
        'Island_locations': frame.Relation_to_UCSC_CpG_Island.map(lambda x: labels(x, 'Unspecified / open sea')),
        'Chromosome': frame.CHR,
        'CpG_coordinate_MAPINFO': frame.MAPINFO,
        'Gene_annotation_raw': frame.UCSC_RefGene_Name.fillna(''),
        'Gene_locations_raw': frame.UCSC_RefGene_Group.fillna(''),
    })
    stem = out / 'data' / f'{disease}_{expected}_CpG_genomic_context'
    table.to_csv(stem.with_suffix('.tsv'), sep='\t', index=False)
    table.to_csv(stem.with_suffix('.csv'), index=False)
    # Markdown is a readable companion; machine-readable exports retain full precision.
    headers = list(table.columns[:8])
    lines = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    for row in table[headers].itertuples(index=False, name=None):
        lines.append('| ' + ' | '.join(f'{x:.6f}' if isinstance(x, float) else str(x) for x in row) + ' |')
    stem.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    metadata = {
        'disease': disease, 'n_CpGs': len(table), 'stage': 'post-evaluation',
        'delta_beta_definition': 'mean disease beta minus mean healthy/control beta; no expression-based filtering',
        'source_columns': {'Delta_beta_external': external, 'Delta_beta_original': original},
        'input_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    }
    stem.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'{disease}: exported all {expected} CpGs -> {stem}.tsv', flush=True)
    return table


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--disease', choices=['AML', 'BC'], required=True)
    args = parser.parse_args()
    run(args.out, args.disease)
