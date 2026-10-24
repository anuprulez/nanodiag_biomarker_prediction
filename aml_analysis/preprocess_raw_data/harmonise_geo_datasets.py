#!/usr/bin/env python3
"""Align shared CpGs and combine healthy/AML beta matrices with bounded memory."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import sqlite3
import tempfile
from geo_common import BASE, matrix_path, read_header, sample_records, beta_rows, valid_betas


def harmonise(data_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = dict(control_probes=0, aml_probes=0, shared_cpgs=0,
                  dropped_missing=0, retained_cpgs=0)
    # Store the smaller cohort on disk; align by identifiers, never row positions.
    with tempfile.TemporaryDirectory(prefix='aml_harmonise_', dir=output_dir) as tmp:
        db = sqlite3.connect(str(Path(tmp) / 'aml.sqlite'))
        try:
            db.execute('CREATE TABLE aml (probe TEXT PRIMARY KEY, beta TEXT NOT NULL)')
            with gzip.open(matrix_path(data_dir, 'GSE58477'), 'rt') as handle:
                samples, metadata = read_header(handle)
                aml_records = sample_records(samples, metadata, 'GSE58477')
                indices = [i for i, row in enumerate(aml_records) if row['selected'] == 'true']
                if not indices:
                    raise ValueError('No leukemic-blast samples found')
                for probe, values in beta_rows(handle, len(samples)):
                    selected = valid_betas([values[i] for i in indices])
                    db.execute('INSERT INTO aml VALUES (?, ?)',
                               (probe, json.dumps(selected)))
                    counts['aml_probes'] += 1
            db.commit()
            with gzip.open(matrix_path(data_dir, 'GSE50660'), 'rt') as handle:
                samples, metadata = read_header(handle)
                controls = sample_records(samples, metadata, 'GSE50660')
                if not controls:
                    raise ValueError('No healthy control samples found')
                selected_aml = [row for row in aml_records if row['selected'] == 'true']
                selected_records = controls + selected_aml
                ids = [row['sample_id'] for row in selected_records]
                if len(ids) != len(set(ids)):
                    raise ValueError('Sample identifiers overlap across studies')
                staged = Path(tmp) / 'combined_beta.tsv.gz'
                with gzip.open(staged, 'wt', newline='', compresslevel=1) as out:
                    writer = csv.writer(out, delimiter='\t', lineterminator='\n')
                    writer.writerow(['ID_REF'] + ids)
                    for probe, values in beta_rows(handle, len(samples)):
                        counts['control_probes'] += 1
                        control_beta = valid_betas(values)
                        match = db.execute('SELECT beta FROM aml WHERE probe=?', (probe,)).fetchone()
                        if match is None or not probe.startswith('cg'):
                            continue
                        counts['shared_cpgs'] += 1
                        combined = control_beta + json.loads(match[0])
                        if None in combined:
                            counts['dropped_missing'] += 1
                            continue
                        writer.writerow([probe] + combined)
                        counts['retained_cpgs'] += 1
            if counts['retained_cpgs'] == 0:
                raise ValueError('No complete shared CpGs remain')
            records = controls + aml_records
            fields = ['sample_id', 'series', 'group', 'label', 'selected']
            fields += sorted(set().union(*(set(row) for row in records)) - set(fields))
            for filename, rows in [('sample_metadata.tsv', selected_records),
                                   ('all_sample_metadata.tsv', records)]:
                with (Path(tmp) / filename).open('w', newline='') as out:
                    writer = csv.DictWriter(out, fieldnames=fields, delimiter='\t', lineterminator='\n')
                    writer.writeheader()
                    writer.writerows(rows)
            counts.update(healthy_samples=len(controls), aml_samples=len(selected_aml),
                          excluded_gse58477_samples=len(aml_records) - len(selected_aml),
                          missing_policy='drop CpGs missing in any selected sample',
                          normalisation='published GEO beta values; no additional normalisation',
                          batch_correction='none; study is confounded with disease group')
            (Path(tmp) / 'harmonisation_report.json').write_text(json.dumps(counts, indent=2) + '\n')
            for filename in ('combined_beta.tsv.gz', 'sample_metadata.tsv',
                             'all_sample_metadata.tsv', 'harmonisation_report.json'):
                (Path(tmp) / filename).replace(output_dir / filename)
        finally:
            db.close()
    print(json.dumps(counts, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=BASE / 'raw/raw')
    parser.add_argument('--output-dir', type=Path, default=BASE / 'raw/harmonised')
    args = parser.parse_args()
    harmonise(args.data_dir, args.output_dir)


if __name__ == '__main__':
    main()
