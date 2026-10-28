#!/usr/bin/env python3
"""Freeze a validation-selected checkpoint and evaluate its existing test partition once."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from tune_models import HERE, GNN_MODELS, seed_all, write_json
from benchmark_common import scores


def checkpoint_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_predictor(study, config, output):
    """Return split predictions without fitting or selecting another checkpoint."""
    name = str(config.model_type)
    if name not in GNN_MODELS:
        import joblib
        from artifact_layout import apply_artifact_layout
        from train_tabular import load_training_dataset
        from benchmark_tabular import predict
        prepared = copy.deepcopy(config)
        prepared.p_generated_files = str(output / 'prepared') + '/'
        apply_artifact_layout(prepared)
        dataset = load_training_dataset(prepared)
        artifact = joblib.load(study / 'best_model.joblib')
        if list(artifact['columns']) != list(dataset.columns):
            raise ValueError('Saved model feature columns differ from current data')
        x, _ = dataset.features(artifact['preprocessing'])
        def run(split):
            seed_all(int(config.SEED))
            mask = dataset.mask(split)
            prediction, probabilities = predict(artifact['model'], x[mask])
            return np.flatnonzero(mask), dataset.y[mask], prediction, probabilities
        try:
            from sklearn.utils import get_tags
            run.non_deterministic = bool(get_tags(artifact['model']).non_deterministic)
        except ImportError:
            getter = getattr(artifact['model'], '_get_tags', None)
            run.non_deterministic = bool(getter().get('non_deterministic', False)) if getter else False
        return run, study / 'best_model.joblib'

    import torch
    from torch_geometric.loader import NeighborLoader
    from utils import choose_model
    # Keep the actual saved filename, including legacy aml/bc naming.
    filename = Path(config.p_torch_data).name
    graph = study / 'prepared' / 'models' / filename
    if not graph.is_file():
        graph = study / 'prepared' / filename
    data = torch.load(graph, map_location='cpu', weights_only=False)
    device = torch.device(config.get('hpo_device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    seed_all(int(config.SEED))
    model = choose_model(config, data, name, False).to(device)
    checkpoint = study / 'best_model.pt'
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
    model.eval()
    def run(split):
        seed_all(int(config.SEED))
        loader = NeighborLoader(data, input_nodes=getattr(data, split + '_mask'),
                                num_neighbors=list(config.neighbors_spread),
                                batch_size=int(config.batch_size), shuffle=False,
                                num_workers=int(config.get('num_workers', 0)),
                                subgraph_type='bidirectional')
        ids, truth, probabilities = [], [], []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                count = int(batch.batch_size)
                logits = model(batch.x, batch.edge_index)[0][:count]
                ids.extend(batch.n_id[:count].cpu().tolist())
                truth.extend(batch.y[:count].cpu().tolist())
                probabilities.extend(logits.softmax(-1).cpu().tolist())
        ids, truth, probabilities = np.asarray(ids), np.asarray(truth), np.asarray(probabilities)
        expected = getattr(data, split + '_mask').nonzero().view(-1).numpy()
        if len(set(ids)) != len(ids) or set(ids) != set(expected):
            raise ValueError('Predicted node IDs do not match requested partition')
        if not np.array_equal(truth, data.y[ids].numpy()):
            raise ValueError('Prediction IDs and labels are misaligned')
        return ids, truth, probabilities.argmax(1), probabilities
    return run, checkpoint


def evaluate_candidate(candidate_path, output):
    candidate = json.loads(Path(candidate_path).read_text())
    if candidate.get('validation_metric') != 'accuracy':
        raise ValueError('Candidate must be selected on validation accuracy')
    study = Path(candidate['study']).resolve()
    config = OmegaConf.load(study / 'best_config.yaml')
    winner = json.loads((study / 'best_params.json').read_text())
    if (winner['model'] != candidate['model'] or winner['trial'] != candidate['trial'] or
            winner['validation_score'] != candidate['validation_score'] or
            winner['input_fingerprint'] != candidate['input_fingerprint']):
        raise ValueError('Candidate is stale or does not match the exported checkpoint')
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output already exists; do not repeat test evaluation')
    output.mkdir(parents=True, exist_ok=True)
    predictor, checkpoint = load_predictor(study, config, output)
    digest = checkpoint_hash(checkpoint)
    source_checkpoint = study / f'trial_{candidate["trial"]:05d}' / ('model.pt' if config.model_type in GNN_MODELS else 'model.joblib')
    source_digest = checkpoint_hash(source_checkpoint) if source_checkpoint.exists() else None
    if source_digest is not None and source_digest != digest:
        raise ValueError('Exported checkpoint differs from the selected trial artifact')
    _, truth, prediction, _ = predictor('val')
    validation = scores(truth, prediction)
    exact = bool(np.isclose(validation['accuracy'], candidate['validation_score'], rtol=0, atol=1e-10))
    nondeterministic = bool(getattr(predictor, 'non_deterministic', False))
    if not exact and not nondeterministic:
        raise ValueError(f'Frozen validation differs: {validation["accuracy"]} vs {candidate["validation_score"]}')
    reproduction = dict(selected_accuracy=candidate['validation_score'],
                        reproduced_accuracy=validation['accuracy'], exact=exact,
                        estimator_declares_nondeterministic=nondeterministic,
                        policy='checkpoint_identity_for_declared_nondeterministic_estimator' if nondeterministic else 'exact')
    frozen = dict(candidate, validation=validation, checkpoint_sha256=digest,
                  source_trial_checkpoint_sha256=source_digest, validation_reproduction=reproduction,
                  selection='Frozen by validation accuracy before test prediction', test_evaluated=False)
    write_json(output / 'selection_frozen_before_test.json', frozen)
    # No test prediction is allowed before the frozen validation/checkpoint record.
    ids, truth, prediction, probabilities = predictor('test')
    test = scores(truth, prediction)
    frame = pd.DataFrame(dict(node_id=ids, true=truth, predicted=prediction))
    for cls in range(probabilities.shape[1]):
        frame[f'probability_{cls}'] = probabilities[:, cls]
    frame.to_csv(output / 'test_predictions.csv', index=False)
    result = dict(frozen, test=test, test_evaluated=True, test_nodes=len(ids),
                  model_settings_installed=False)
    write_json(output / 'result.json', result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    candidate = args.candidate.resolve()
    output = args.output.resolve()
    os.chdir(HERE)
    evaluate_candidate(candidate, output)


if __name__ == '__main__':
    main()
