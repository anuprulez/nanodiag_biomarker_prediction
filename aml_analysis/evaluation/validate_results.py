#!/usr/bin/env python3
"""Validate all output rows and independently recompute three raw CpGs."""
import csv
import gzip
import json
import math
from pathlib import Path
import statistics
import sys

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent / 'preprocess_raw_data'))
from geo_common import read_header, beta_rows


def main():
    report = json.loads((BASE / 'results/analysis_report.json').read_text())
    pairs, probes, checks = set(), set(), {}
    with (BASE / 'results/cpg_gene_beta_comparison.tsv').open() as source:
        for row in csv.DictReader(source, delimiter='\t'):
            key = row['CpG_Gene']
            assert key == row['CpG_ID'] + '_' + row['Gene']
            assert key not in pairs and row['Gene'] and ';' not in row['Gene']
            pairs.add(key)
            probes.add(row['CpG_ID'])
            h = row['Healthy_GSE40279_mean_beta']
            a = row['AML_GSE62298_mean_beta']
            if h and a:
                assert 0 <= float(h) <= 1 and 0 <= float(a) <= 1
                assert math.isclose(float(a) - float(h), float(row['Delta_beta_AML_minus_healthy']), abs_tol=2e-12)
            else:
                assert row['Delta_beta_AML_minus_healthy'] == ''
            for prefix, size in [('Healthy_GSE40279', 656), ('AML_GSE62298', 68)]:
                n = int(row[prefix + '_n'])
                assert 0 <= n <= size
                assert (n == 0) == (row[prefix + '_mean_beta'] == '')
                assert (n < 2) == (row[prefix + '_SD'] == '')
                if n > 1:
                    assert float(row[prefix + '_SD']) >= 0
            if len(checks) < 3 and h and a:
                checks.setdefault(row['CpG_ID'], row)
    assert len(pairs) == report['cpg_gene_rows']
    assert len(probes) == report['shared_gene_annotated_cpgs']
    with (BASE / 'results/shared_cpg_statistics.tsv').open() as source:
        all_probes = [row['CpG_ID'] for row in csv.DictReader(source, delimiter='\t')]
    assert len(set(all_probes)) == len(all_probes) == report['shared_cpgs']
    for series, prefix in [('GSE40279', 'Healthy_GSE40279'), ('GSE62298', 'AML_GSE62298')]:
        remaining = set(checks)
        with gzip.open(BASE / 'raw' / f'{series}_series_matrix.txt.gz', 'rt') as source:
            samples, _ = read_header(source)
            for probe, values in beta_rows(source, len(samples)):
                if probe not in remaining:
                    continue
                values = [float(x) for x in values if x.lower() not in ('na', 'nan', 'null', 'n/a', '')]
                row = checks[probe]
                assert len(values) == int(row[prefix + '_n'])
                assert math.isclose(statistics.mean(values), float(row[prefix + '_mean_beta']), abs_tol=1e-12)
                assert math.isclose(statistics.stdev(values), float(row[prefix + '_SD']), abs_tol=1e-12)
                remaining.remove(probe)
                if not remaining:
                    break
        assert not remaining
    result = {'passed': True, 'cpg_gene_rows_checked': len(pairs), 'shared_cpgs_checked': len(all_probes),
              'independently_recomputed_cpgs_per_dataset': sorted(checks)}
    (BASE / 'results/validation_report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
