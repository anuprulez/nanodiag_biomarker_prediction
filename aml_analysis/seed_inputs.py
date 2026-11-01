"""Shared AML seed parsing and preprocessing provenance."""
import csv
import hashlib
import json
import math
from pathlib import Path

DELTA_COLUMN = 'deltaBeta_AML_minus_Normal'


def read_seed_rows(path):
    with Path(path).open(newline='') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        required = {'CpG_ID', 'Gene', DELTA_COLUMN, 'CpG_Gene'}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f'AML seed file requires columns {sorted(required)}')
        rows = list(reader)
    seen = set()
    if not rows:
        raise ValueError('AML seed file is empty')
    for row in rows:
        probe = row['CpG_ID'].strip()
        delta = float(row[DELTA_COLUMN])
        if not probe.startswith('cg') or not probe[2:].isdigit() or probe in seen:
            raise ValueError(f'Invalid or duplicate seed CpG: {probe}')
        if not row['Gene'].strip() or not math.isfinite(delta) or not -1 <= delta <= 1:
            raise ValueError(f'Missing gene or invalid delta beta for {probe}')
        seen.add(probe)
    return rows


def seed_signature(path):
    path = Path(path).resolve()
    rows = read_seed_rows(path)
    return {'seed_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'seed_count': len(rows), 'delta_column': DELTA_COLUMN}


def write_seed_manifest(config, expected):
    current = seed_signature(config.p_seeds)
    if current != expected:
        raise ValueError('Seed file changed during preprocessing; rerun preprocessing')
    target = Path(config.p_preprocessing_manifest)
    temporary = target.with_suffix('.json.part')
    temporary.write_text(json.dumps(dict(current, seed_path=str(Path(config.p_seeds).resolve())), indent=2) + '\n')
    temporary.replace(target)


def validate_seed_manifest(config):
    path = Path(config.p_preprocessing_manifest)
    if not path.exists():
        raise ValueError('Preprocessing has not completed for the new seed workflow. Run run_pipeline.py --stage preprocess first.')
    recorded = json.loads(path.read_text())
    if any(recorded.get(key) != value for key, value in seed_signature(config.p_seeds).items()):
        raise ValueError('Preprocessing outputs use a different seed file. Rerun preprocessing before training.')
