#!/usr/bin/env python3
"""Assess or apply GPL13534 RefGene annotations to the combined beta matrix."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
from geo_common import BASE

SOURCE_URL = 'https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL13nnn/GPL13534/soft/GPL13534_family.soft.gz'


def load_annotation(soft, cache):
    if not cache.exists():
        temporary = cache.with_suffix('.tsv.part')
        with gzip.open(soft, 'rt') as handle, temporary.open('w', newline='') as out:
            for line in handle:
                if line.strip() == '!platform_table_begin':
                    break
            else:
                raise ValueError('Platform table missing')
            reader = csv.DictReader(handle, delimiter='\t')
            fields = ['ID', 'UCSC_RefGene_Name', 'CHR', 'MAPINFO']
            if not set(fields) <= set(reader.fieldnames):
                raise ValueError('Required platform columns missing')
            writer = csv.DictWriter(out, fieldnames=fields, delimiter='\t', lineterminator='\n')
            writer.writeheader()
            for row in reader:
                if row['ID'] == '!platform_table_end':
                    break
                writer.writerow({key: row[key] for key in fields})
            else:
                raise ValueError('Incomplete platform table')
        temporary.replace(cache)
    annotations = {}
    with cache.open() as handle:
        for row in csv.DictReader(handle, delimiter='\t'):
            probe = row['ID']
            if probe in annotations:
                raise ValueError(f'Duplicate platform probe {probe}')
            row['genes'] = list(dict.fromkeys(g.strip() for g in row['UCSC_RefGene_Name'].split(';') if g.strip()))
            annotations[probe] = row
    return annotations


def run(args):
    annotations = load_annotation(args.platform, args.annotation_cache)
    report = dict(output_rows=0, total_cpgs=0, single_gene=0, multiple_genes=0, unannotated=0, absent_from_platform=0)
    with gzip.open(args.input, 'rt') as handle:
        header = next(handle).rstrip('\n').split('\t')
        if header[0] != 'ID_REF':
            raise ValueError('Expected original CpG-only matrix with ID_REF header')
        report['samples'] = len(header) - 1
        for line in handle:
            probe = line.split('\t', 1)[0]
            genes = annotations.get(probe, {}).get('genes', [])
            report['total_cpgs'] += 1
            report['output_rows'] += len(genes)
            if probe not in annotations:
                report['absent_from_platform'] += 1
            report['single_gene' if len(genes) == 1 else 'multiple_genes' if genes else 'unannotated'] += 1
    report.update(source_url=SOURCE_URL, annotation='GPL13534 UCSC_RefGene_Name (legacy hg19 annotation)',
                  multi_gene_policy='deduplicate gene symbols and emit one row per CpG-gene pair',
                  missing_gene_policy='drop CpGs without RefGene annotation')
    print(json.dumps(report, indent=2), flush=True)
    if not args.apply:
        return
    if args.input.resolve() == args.output.resolve():
        raise ValueError('Use a separate output to preserve the original matrix')
    temporary = args.output.with_suffix(args.output.suffix + '.part')
    mapping = args.output.parent / 'cpg_gene_mapping.tsv'
    mapping_tmp = mapping.with_suffix('.tsv.part')
    before = hashlib.sha256()
    seen = set()
    with gzip.open(args.input, 'rt') as source, gzip.open(temporary, 'wt', compresslevel=1) as target, mapping_tmp.open('w', newline='') as out:
        header = next(source)
        target.write(header)  # ID_REF stays compatible with existing matrix readers.
        writer = csv.writer(out, delimiter='\t', lineterminator='\n')
        writer.writerow(['CpG_ID', 'Gene', 'CpG_Gene', 'CHR', 'MAPINFO_hg19', 'annotation_status'])
        for line in source:
            probe, separator, values = line.partition('\t')
            if not separator or probe in seen:
                raise ValueError(f'Malformed or duplicate matrix probe {probe}')
            seen.add(probe)
            annotation = annotations.get(probe, {})
            for gene in annotation.get('genes', []):
                identifier = probe + '_' + gene
                target.write(identifier + '\t' + values)
                before.update(values.encode())
                writer.writerow([probe, gene, identifier, annotation.get('CHR', ''), annotation.get('MAPINFO', ''),
                                 'gene_associated'])
    # Independently read the full output; verify row identities and beta text.
    after = hashlib.sha256()
    count = 0
    identifiers = set()
    with gzip.open(temporary, 'rt') as handle, mapping_tmp.open() as out:
        if next(handle) != header:
            raise ValueError('Sample header changed')
        mapped = csv.DictReader(out, delimiter='\t')
        for line in handle:
            identifier, _, values = line.partition('\t')
            row = next(mapped)
            if identifier in identifiers or ';' in identifier or identifier.endswith('_UNANNOTATED'):
                raise ValueError('Duplicate, unexpanded or unannotated identifier')
            identifiers.add(identifier)
            if identifier != row['CpG_Gene'] or values.count('\t') != report['samples'] - 1:
                raise ValueError('Output mapping or dimensions mismatch')
            after.update(values.encode())
            count += 1
        if next(mapped, None) is not None:
            raise ValueError('Extra mapping rows')
    if before.digest() != after.digest() or count != report['output_rows']:
        raise ValueError('Beta values or row count changed')
    report['beta_values_sha256'] = before.hexdigest()
    report['verified_beta_values_unchanged'] = True
    temporary.replace(args.output)
    mapping_tmp.replace(mapping)
    (args.output.parent / 'cpg_gene_annotation_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Wrote and verified {args.output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--platform', type=Path, default=BASE / 'data/GPL13534_family.soft.gz')
    parser.add_argument('--annotation-cache', type=Path, default=BASE / 'data/GPL13534_probe_annotation.tsv')
    parser.add_argument('--input', type=Path, default=BASE / 'data/harmonised/combined_beta.tsv.gz')
    parser.add_argument('--output', type=Path, default=BASE / 'data/harmonised/combined_beta_cpg_gene.tsv.gz')
    parser.add_argument('--apply', action='store_true', help='Write annotated matrix after assessment')
    run(parser.parse_args())
