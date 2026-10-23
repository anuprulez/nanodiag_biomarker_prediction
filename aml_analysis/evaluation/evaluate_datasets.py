#!/usr/bin/env python3
"""Compare published GEO beta values at shared CpGs; standard library only."""
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent / 'preprocess_raw_data'))
from geo_common import read_header, beta_rows
from annotate_combined_beta import load_annotation, SOURCE_URL

SERIES = ('GSE40279', 'GSE62298')


def summarize(series):
    path = BASE / 'raw' / f'{series}_series_matrix.txt.gz'
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    stats = {}
    report = {'rows': 0, 'non_cpg_rows': 0, 'missing_values': 0, 'all_missing_cpgs': 0}
    with gzip.open(path, 'rt') as source:
        samples, metadata = read_header(source)
        assert metadata['!Sample_platform_id'] == [['GPL13534'] * len(samples)]
        if series == 'GSE62298':
            assert all(x == 'Primary AML patient' for x in metadata['!Sample_source_name_ch1'][0])
        else:
            assert ['tissue: whole blood'] * len(samples) in metadata['!Sample_characteristics_ch1']
        (BASE / 'raw' / f'{series}_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        with (BASE / 'results' / f'{series}_samples.tsv').open('w') as out:
            writer = csv.writer(out, delimiter='\t', lineterminator='\n')
            writer.writerow(['sample_id', 'series', 'group', 'title', 'characteristics'])
            for i, sample in enumerate(samples):
                writer.writerow([sample, series, 'healthy' if series == SERIES[0] else 'AML',
                                 metadata['!Sample_title'][0][i],
                                 ' | '.join(row[i] for row in metadata['!Sample_characteristics_ch1'])])
        for probe, values in beta_rows(source, len(samples)):
            report['rows'] += 1
            if not probe.startswith('cg'):
                report['non_cpg_rows'] += 1
                continue
            nums = []
            for value in values:
                if value.strip().lower() in ('', 'na', 'nan', 'null', 'n/a'):
                    report['missing_values'] += 1
                    continue
                number = float(value)
                if not math.isfinite(number) or not 0 <= number <= 1:
                    raise ValueError(f'{series} {probe}: invalid beta {value}')
                nums.append(number)
            n = len(nums)
            mean = math.fsum(nums) / n if n else None
            sd = math.sqrt(math.fsum((x - mean) ** 2 for x in nums) / (n - 1)) if n > 1 else None
            stats[probe] = (mean, sd, n)
            report['all_missing_cpgs'] += n == 0
            if report['rows'] % 100000 == 0:
                print(f'{series}: {report["rows"]:,} rows processed', flush=True)
        # Consume through EOF to validate gzip trailer/CRC.
        source.read()
    report.update(samples=len(samples), cpgs=len(stats), sha256=digest.hexdigest(), bytes=path.stat().st_size,
                  url=f'https://ftp.ncbi.nlm.nih.gov/geo/series/{series[:-3]}nnn/{series}/matrix/{path.name}')
    print(f'{series}: {report}', flush=True)
    return stats, report


def main():
    (BASE / 'results').mkdir(exist_ok=True)
    healthy, healthy_report = summarize(SERIES[0])
    aml, aml_report = summarize(SERIES[1])
    annotations = load_annotation(BASE.parent / 'raw/GPL13534_family.soft.gz',
                                  BASE.parent / 'raw/GPL13534_probe_annotation.tsv')
    shared = sorted(healthy.keys() & aml.keys())
    fields = ['CpG_ID', 'Gene', 'CpG_Gene', 'Healthy_GSE40279_mean_beta', 'Healthy_GSE40279_SD',
              'Healthy_GSE40279_n', 'AML_GSE62298_mean_beta', 'AML_GSE62298_SD', 'AML_GSE62298_n',
              'Delta_beta_AML_minus_healthy', 'CHR', 'MAPINFO_hg19']
    count = unannotated = 0
    preview = []
    with (BASE / 'results/cpg_gene_beta_comparison.tsv').open('w') as out, \
            (BASE / 'results/shared_cpg_statistics.tsv').open('w') as all_out:
        writer = csv.writer(out, delimiter='\t', lineterminator='\n')
        all_writer = csv.writer(all_out, delimiter='\t', lineterminator='\n')
        writer.writerow(fields)
        all_writer.writerow([fields[0]] + fields[3:10] + ['Genes', 'annotation_status'])
        for probe in shared:
            h, a = healthy[probe], aml[probe]
            delta = a[0] - h[0] if a[0] is not None and h[0] is not None else None
            values = [*h, *a, delta]
            values = ['' if x is None else format(x, '.12g') if isinstance(x, float) else x for x in values]
            ann = annotations.get(probe, {})
            genes = ann.get('genes', [])
            unannotated += not genes
            all_writer.writerow([probe, *values, ';'.join(genes), 'gene_associated' if genes else 'unannotated'])
            for gene in genes:
                row = [probe, gene, probe + '_' + gene, *values, ann.get('CHR', ''), ann.get('MAPINFO', '')]
                writer.writerow(row)
                count += 1
                if len(preview) < 10 and delta is not None:
                    preview.append((probe + '_' + gene, h[0], h[1], a[0], a[1], delta))
    report = {'datasets': dict(zip(SERIES, [healthy_report, aml_report])), 'shared_cpgs': len(shared),
              'healthy_only_cpgs': len(healthy.keys() - aml.keys()), 'aml_only_cpgs': len(aml.keys() - healthy.keys()),
              'shared_gene_annotated_cpgs': len(shared) - unannotated, 'shared_unannotated_cpgs': unannotated,
              'cpg_gene_rows': count, 'annotation_source': SOURCE_URL,
              'annotation_sha256': hashlib.sha256((BASE.parent / 'raw/GPL13534_probe_annotation.tsv').read_bytes()).hexdigest(),
              'SD': 'sample standard deviation, denominator n-1', 'delta': 'mean(GSE62298) - mean(GSE40279)',
              'missing_policy': 'omit missing beta values per CpG; no imputation; undefined statistics blank',
              'gene_policy': 'one row per unique RefGene symbol per CpG; unannotated retained in shared_cpg_statistics.tsv',
              'harmonization': 'exact intersection of cg probe IDs on GPL13534; published beta values unchanged',
              'limitations': 'Descriptive cross-study comparison; no batch, age, sex or cell-composition adjustment. GSE62298 is GEO, not TCGA.'}
    (BASE / 'results/analysis_report.json').write_text(json.dumps(report, indent=2) + '\n')
    lines = ['# GSE40279 versus GSE62298', '',
             f'Healthy: {healthy_report["samples"]} samples. AML: {aml_report["samples"]} samples.',
             f'Shared CpGs: {len(shared):,}; CpG–gene rows: {count:,}.', '',
             'Full table: `results/cpg_gene_beta_comparison.tsv`. All shared CpGs, including unannotated probes: `results/shared_cpg_statistics.tsv`.', '',
             'The preview below contains the first 10 CpG–gene pairs in CpG ID order, not a significance ranking.', '',
             '| CpG_gene | Healthy GSE40279 (mean ± SD) | AML GSE62298 (mean ± SD) | Δβ (AML − healthy) |',
             '|---|---:|---:|---:|']
    for key, hm, hs, am, ass, d in preview:
        lines.append(f'| {key} | {hm:.6f} ± {hs:.6f} | {am:.6f} ± {ass:.6f} | {d:+.6f} |')
    lines += ['', 'From aml_analysis, run `python evaluation/download_datasets.py` to download/validate the matrices, then `python evaluation/evaluate_datasets.py` to regenerate the results.', '',
              'Method: exact intersection of cg probe IDs; GPL13534 UCSC_RefGene_Name (legacy hg19) annotation reused from ../raw. Multiple gene assignments are expanded into unique CpG–gene pairs. Published beta values are unchanged. SD uses n−1; missing values are omitted per probe, with valid n reported. No imputation, filtering by effect, or statistical significance threshold was applied.', '',
              'These are descriptive cross-study differences without batch, age, sex, or cell-composition adjustment. Whole blood controls and primary AML blood differ in cellular composition. GSE62298 is the requested AML GEO dataset; it is not TCGA.', '',
              'Dataset URLs, checksums, row counts and policies are in results/analysis_report.json; sample metadata and cohort tables are retained in raw/ and results/.', '']
    (BASE / 'README.md').write_text('\n'.join(lines))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
