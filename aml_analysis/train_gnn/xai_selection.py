"""Select XAI targets from LP/RN exports without importing the training stack."""
import csv
import math
from pathlib import Path


def add_xai_selection_arguments(parser):
    parser.add_argument('--xai-group', '--prediction-group', dest='xai_group',
                        type=str.upper, choices=('LP', 'RN'), default='LP',
                        help='XAI prediction file: LP=likely positive (default); RN=predicted negatives')
    parser.add_argument('--top-k', type=int, default=1,
                        help='Explain this many highest-probability predictions when no node ID/name is supplied (default: 1)')


def prediction_path(config, group='LP'):
    group = group.upper()
    if group not in ('LP', 'RN'):
        raise ValueError('XAI prediction group must be LP or RN')
    return Path(config.p_pred_likely_pos if group == 'LP' else config.p_pred_negatives)


def top_prediction_nodes(path, top_k=1):
    """Rank by pred_probs descending, preserving file order for tied scores."""
    if top_k < 1:
        raise ValueError('--top-k must be at least 1')
    path = Path(path)
    with path.open(newline='') as source:
        header = source.readline()
        source.seek(0)
        reader = csv.DictReader(source, delimiter='\t' if '\t' in header else ',')
        if not {'test_gene_ids', 'pred_probs'} <= set(reader.fieldnames or []):
            raise ValueError(f'{path}: require test_gene_ids and pred_probs columns')
        rows = []
        seen = set()
        for row in reader:
            node = int(row['test_gene_ids'])
            score = float(row['pred_probs'])
            if node < 0 or node in seen:
                raise ValueError(f'{path}: negative or duplicate node ID {node}')
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f'{path}: invalid prediction probability for node {node}')
            seen.add(node)
            rows.append((node, score))
    if not rows:
        raise ValueError(f'{path}: no predicted nodes available for XAI')
    if len(rows) < top_k:
        raise ValueError(f'{path}: requested top {top_k} nodes but only {len(rows)} available')
    return [node for node, _ in sorted(rows, key=lambda row: -row[1])[:top_k]]


def select_xai_targets(config, node_id=None, node_name=None, group='LP', top_k=1,
                       refresh_predictions=None):
    """Explicit IDs/names bypass prediction-file lookup."""
    if node_id is not None:
        if node_id < 0:
            raise ValueError('--node-id must be nonnegative')
        return [(node_id, None)]
    if node_name is not None:
        return [(None, node_name)]
    path = prediction_path(config, group)
    if not path.is_file() and refresh_predictions is not None:
        refresh_predictions(config)
    return [(node, None) for node in top_prediction_nodes(path, top_k)]
