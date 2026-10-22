"""Streaming GEO Series Matrix readers (Python standard library only)."""
import csv
import gzip
import math
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
SERIES = ('GSE50660', 'GSE58477')


def matrix_path(directory, series):
    return Path(directory) / f'{series}_series_matrix.txt.gz'


def read_header(handle):
    metadata = {}
    for line in handle:
        fields = next(csv.reader([line], delimiter='\t'))
        if not fields:
            continue
        if fields[0] == '!series_matrix_table_begin':
            header = next(csv.reader([next(handle)], delimiter='\t'))
            if header[0] != 'ID_REF' or len(header) < 2:
                raise ValueError('Missing Series Matrix sample columns')
            samples = header[1:]
            if len(samples) != len(set(samples)):
                raise ValueError('Duplicate sample identifiers')
            if metadata.get('!Sample_geo_accession') != [samples]:
                raise ValueError('Metadata sample order differs from matrix')
            for values in metadata.values():
                if any(len(row) != len(samples) for row in values):
                    raise ValueError('Metadata column count differs from matrix')
            return samples, metadata
        if fields[0].startswith('!Sample_'):
            metadata.setdefault(fields[0], []).append(fields[1:])
    raise ValueError('Series Matrix table missing')


def sample_records(samples, metadata, series):
    records = []
    for i, sample in enumerate(samples):
        record = {'sample_id': sample, 'series': series}
        for key, values in metadata.items():
            if key == '!Sample_characteristics_ch1':
                for row in values:
                    name, sep, value = row[i].partition(': ')
                    if sep:
                        record['characteristic:' + name] = value
            else:
                record[key.removeprefix('!Sample_')] = ' | '.join(row[i] for row in values)
        if record.get('platform_id') != 'GPL13534':
            raise ValueError(f'{sample}: expected platform GPL13534')
        if series == 'GSE50660':
            if record.get('source_name_ch1', '').lower() != 'peripheral blood':
                raise ValueError(f'{sample}: unexpected control tissue')
            record.update(group='healthy_control', label='0', selected='true')
        else:
            cell = record.get('characteristic:cell type', '').lower()
            if cell not in ('leukemic blast', 'normal bone marrow progenitor cells'):
                raise ValueError(f'{sample}: unknown cell type {cell!r}')
            selected = cell == 'leukemic blast'
            record.update(group='AML' if selected else 'excluded_healthy_bone_marrow',
                          label='1' if selected else '', selected=str(selected).lower())
        records.append(record)
    return records


def beta_rows(handle, width):
    seen = set()
    for row in csv.reader(handle, delimiter='\t'):
        if row and row[0] == '!series_matrix_table_end':
            return
        if len(row) != width + 1:
            raise ValueError('Matrix row has incorrect number of samples')
        probe = row[0]
        if not probe or probe in seen:
            raise ValueError(f'Empty or duplicate probe identifier: {probe}')
        seen.add(probe)
        yield probe, row[1:]
    raise ValueError('Truncated matrix: missing table end marker')


def valid_betas(values):
    """Return None for missing data; reject non-beta numeric measurements."""
    result = []
    for value in values:
        if value.strip().lower() in ('', 'na', 'nan', 'null', 'n/a'):
            result.append(None)
            continue
        number = float(value)
        if not math.isfinite(number) or not 0 <= number <= 1:
            raise ValueError(f'Expected beta in [0, 1], found {value!r}')
        result.append(value)
    return result
