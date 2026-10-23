"""Validated streaming readers and source definitions for external BC evaluation."""
import csv
import gzip
import hashlib
import math
from pathlib import Path
import re

BASE = Path(__file__).resolve().parent
SERIES = ('GSE101961', 'GSE226569')  # healthy, breast cancer
PLATFORM_URL = 'https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL13nnn/GPL13534/soft/GPL13534_family.soft.gz'
SOURCES = {
    f'{s}_series_matrix.txt.gz': f'https://ftp.ncbi.nlm.nih.gov/geo/series/{s[:-3]}nnn/{s}/matrix/{s}_series_matrix.txt.gz'
    for s in SERIES
}
SOURCES['GSE226569_Methy_ProcessedMatrix.csv.gz'] = 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE226nnn/GSE226569/suppl/GSE226569_Methy_ProcessedMatrix.csv.gz'
MISSING = {'', 'na', 'nan', 'null', 'n/a'}


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


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
            if len(samples) != len(set(samples)) or metadata.get('!Sample_geo_accession') != [samples]:
                raise ValueError('Duplicate samples or metadata/matrix sample order mismatch')
            if any(len(row) != len(samples) for rows in metadata.values() for row in rows):
                raise ValueError('Metadata column count differs from matrix')
            return samples, metadata
        if fields[0].startswith('!Sample_'):
            metadata.setdefault(fields[0], []).append(fields[1:])
    raise ValueError('Series Matrix table missing')


def beta_rows(handle, width):
    seen = set()
    for row in csv.reader(handle, delimiter='\t'):
        if row and row[0] == '!series_matrix_table_end':
            return
        if len(row) != width + 1 or not row[0] or row[0] in seen:
            raise ValueError('Invalid matrix width, empty or duplicate probe')
        seen.add(row[0])
        yield row[0], row[1:]
    raise ValueError('Truncated Series Matrix')


def metadata(series, raw):
    with gzip.open(raw / f'{series}_series_matrix.txt.gz', 'rt') as source:
        samples, meta = read_header(source)
    if meta.get('!Sample_platform_id') != [['GPL13534'] * len(samples)]:
        raise ValueError(f'{series}: expected GPL13534')
    if series == 'GSE226569':
        if meta.get('!Sample_source_name_ch1') != [['FFPE breast tumor tissue'] * len(samples)]:
            raise ValueError('Unexpected tumor tissue')
        if ['tissue: breast tumor'] * len(samples) not in meta.get('!Sample_characteristics_ch1', []):
            raise ValueError('Unexpected tumor characteristics')
    elif series == 'GSE101961':
        if not all(x.startswith('breast tissue ') for x in meta['!Sample_source_name_ch1'][0]):
            raise ValueError('Unexpected control tissue')
    else:
        raise ValueError(f'Unknown cohort {series}')
    return samples, meta


def tumor_columns(header, titles):
    """Select only beta columns, excluding the interleaved detection p-values."""
    if len(header) != len(set(header)) or len(titles) != len(set(titles)):
        raise ValueError('Duplicate supplementary columns or sample titles')
    expected = set(titles)
    betas = header[1::2]
    detections = header[2::2]
    if set(betas) != expected or len(betas) != len(titles) or detections != [s + '.DetectionPvalue' for s in betas]:
        raise ValueError('Supplementary beta/detection columns do not match GEO sample titles')
    positions = {name: i for i, name in enumerate(header)}
    return [positions[name] for name in titles]


def rows(series, raw):
    samples, meta = metadata(series, raw)
    if series == 'GSE101961':
        with gzip.open(raw / f'{series}_series_matrix.txt.gz', 'rt') as source:
            read_header(source)
            yield from beta_rows(source, len(samples))
            source.read()  # validate CRC/trailer
    else:
        with gzip.open(raw / 'GSE226569_Methy_ProcessedMatrix.csv.gz', 'rt') as source:
            reader = csv.reader(source)
            header = next(reader)
            indices = tumor_columns(header, meta['!Sample_title'][0])
            seen = set()
            for row in reader:
                if len(row) != len(header) or not row[0] or row[0] in seen:
                    raise ValueError('Invalid tumor matrix row or duplicate probe')
                seen.add(row[0])
                yield row[0], [row[i] for i in indices]
            if not seen:
                raise ValueError('Empty tumor beta matrix')


def statistics(values):
    nums = []
    for value in values:
        if value.strip().lower() in MISSING:
            continue
        number = float(value)
        if not math.isfinite(number) or not 0 <= number <= 1:
            raise ValueError(f'Invalid beta value {value!r}')
        nums.append(number)
    n = len(nums)
    mean = math.fsum(nums) / n if n else None
    sd = math.sqrt(math.fsum((x-mean)**2 for x in nums)/(n-1)) if n > 1 else None
    return mean, sd, n


def annotations(raw):
    cache = raw / 'GPL13534_probe_annotation.tsv'
    result = {}
    with cache.open() as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        if not {'ID','UCSC_RefGene_Name','CHR','MAPINFO'} <= set(reader.fieldnames or []):
            raise ValueError('Invalid platform annotation columns')
        for row in reader:
            if row['ID'] in result:
                raise ValueError('Duplicate annotated probe')
            row['genes'] = sorted(set(x.strip() for x in row['UCSC_RefGene_Name'].split(';') if x.strip()))
            result[row['ID']] = row
    return result
