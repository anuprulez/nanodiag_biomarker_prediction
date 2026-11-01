"""Matched held-out-node UMAP comparisons from an archived PNA training run."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                            calinski_harabasz_score, adjusted_rand_score)
from sklearn.manifold import trustworthiness
from .analysis import digest
from .publication import STYLE, save_publication

# Match the existing training UMAPs: seaborn tab10, first five class colors.
CLASSES = {1: ('Positive seed', '#1F77B4'), 2: ('Likely positive', '#FF7F0E'),
           3: ('Weakly negative', '#2CA02C'), 4: ('Likely negative', '#D62728'),
           5: ('Reliable negative', '#9467BD')}


def aligned_inputs(features, predictions, rankings, nedbit_columns):
    """Join by node name and independently check saved graph IDs and NIAPU labels."""
    if not features.name.is_unique or not predictions.test_gene_names.is_unique:
        raise ValueError('Feature and prediction node names must be unique')
    if not rankings.node_name.is_unique:
        raise ValueError('NIAPU ranking node names must be unique')
    if set(nedbit_columns) & {'name', 'labels'}:
        raise ValueError('Node identifiers and labels cannot be features')
    ids = predictions.test_gene_ids.to_numpy(int)
    if not np.array_equal(features.iloc[ids].name, predictions.test_gene_names):
        raise ValueError('Archived prediction graph IDs do not match feature node names')
    selected = features.set_index('name').loc[predictions.test_gene_names].copy()
    truth = predictions.labels.to_numpy(int)
    current = rankings.set_index('node_name').loc[selected.index, 'niapu_class'].to_numpy(int)
    if not np.array_equal(selected.labels.to_numpy(int), truth) or not np.array_equal(current, truth):
        raise ValueError('Archived feature, prediction and current NIAPU labels differ')
    pred = predictions.pred_labels.to_numpy(int)
    if not set(truth).issubset(CLASSES) or not set(pred).issubset(CLASSES):
        raise ValueError('Expected NIAPU/PNA classes 1 through 5')
    methylation = [c for c in selected.columns if c not in ['labels', *nedbit_columns]]
    if not methylation or not nedbit_columns:
        raise ValueError('Both NeDBIT and DNA methylation feature blocks are required')
    blocks = []
    info = []
    for columns in [nedbit_columns, methylation]:
        x = selected[columns].to_numpy(float)
        if np.isinf(x).any() or np.isnan(x).all(axis=0).any():
            raise ValueError('Infinite or entirely missing feature column')
        missing = int(np.isnan(x).sum())
        x = SimpleImputer(strategy='median').fit_transform(x)
        x = StandardScaler().fit_transform(x)
        varying = np.ptp(x, axis=0) > 0
        if not varying.any():raise ValueError('Feature block has no variation')
        # Equal total variance per block: hundreds of patient measurements must
        # not outweigh the four graph features just by their column count.
        blocks.append(x[:, varying] / np.sqrt(varying.sum()))
        info.append(dict(columns=list(columns), varying_columns=int(varying.sum()), imputed_values=missing))
    return selected.index.to_numpy(), truth, pred, blocks, info


def quality(x, embedding, labels):
    """Metrics use fixed supplied labels, not clusters inferred from the embedding."""
    valid = 1 < len(np.unique(labels)) < len(labels)
    return dict(silhouette_2d=float(silhouette_score(embedding, labels)) if valid else None,
                silhouette_features=float(silhouette_score(x, labels)) if valid else None,
                davies_bouldin_2d=float(davies_bouldin_score(embedding, labels)) if valid else None,
                calinski_harabasz_2d=float(calinski_harabasz_score(embedding, labels)) if valid else None)


def render(out, disease, run_name, embeddings, truth, pred, metrics):
    titles = ['NIAPU labels, NeDBIT features', 'NIAPU labels, NeDBIT + DNA methylation features',
              'NIAPU labels, DNA methylation features', 'PNA labels, Final PNAConv representation']
    def draw(ax, i):
        e = embeddings[i]
        labels = pred if i == 3 else truth
        for label, (_, color) in CLASSES.items():
            mask = labels == label
            ax.scatter(*e[mask].T, s=15, c=color, alpha=.9, linewidths=0, rasterized=True)
        ax.text(0., 1.055, 'ABCD'[i], transform=ax.transAxes, fontsize=17, fontweight='bold', va='bottom')
        ax.set_title(titles[i], loc='center', pad=15, fontsize=14)
        ax.set_xlabel('UMAP 1', fontsize=14);ax.set_ylabel('UMAP 2', fontsize=14)
        ax.set_xticks([]);ax.set_yticks([])
        ax.spines[['top','right']].set_visible(False)
        ax.margins(.08)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))
        for i, ax in enumerate(axes.flat):draw(ax, i)
        fig.suptitle(f'{disease}: NIAPU and PNA feature-space comparison', fontsize=22, y=.985)
        handles = [Line2D([], [], marker='o', ls='', color=color, markersize=8, label=name)
                   for name, color in CLASSES.values()]
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .012), ncol=5, frameon=False, fontsize=12)
        fig.subplots_adjust(left=.065, right=.985, top=.89, bottom=.11, hspace=.27, wspace=.17)
        save_publication(fig, out, 'niapu_umap_comparison')
        for i in range(4):
            fig, ax = plt.subplots(figsize=(8.8, 7.5));draw(ax, i)
            fig.suptitle(f'{disease}', fontsize=16)
            fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, fontsize=10)
            fig.subplots_adjust(left=.1,right=.97,top=.86,bottom=.15)
            save_publication(fig, out, f'niapu_umap_{i+1}')


def run(root, config, run_name='run1'):
    root = Path(root).resolve();base = Path(config.p_base).resolve()
    archive = base/'runs'/run_name
    key = 'bc' if root.name == 'breast_cancer_analysis' else 'aml'
    disease = 'Breast cancer' if key == 'bc' else 'AML'
    paths = dict(features=archive/f'generated/data/df_nebit_dnam_features_{key}_PNA.csv',
                 predictions=archive/'generated/data/all_test_predictions_PNA.tsv',
                 checkpoint=archive/f'generated/models/best_model_{key}_PNA.pt',
                 archived_config=archive/'config.yaml', run_manifest=archive/'run.json',
                 rankings=Path(config.p_out_gene_rankings))
    for p in paths.values():
        if not p.is_file():raise FileNotFoundError(f'{p}: UMAP needs an archived PNA run; train PNA first or select --run')
    saved = OmegaConf.load(paths['archived_config'])
    columns = [x.strip() for x in saved.keep_feature_names.split(',')]
    features = pd.read_csv(paths['features']);predictions = pd.read_csv(paths['predictions'], sep='\t')
    from .niapu_rankings import load_rankings
    rankings, _, _ = load_rankings(paths['rankings'])
    names, truth, pred, blocks, info = aligned_inputs(features, predictions, rankings, columns)
    # Verify the exported table against the raw labels saved after best-checkpoint
    # restoration. No PNA refitting, surrogate classifier or in-sample predictions.
    import torch
    for field, values in [('pred_labels', pred), ('true_labels', truth), ('test_ids', predictions.test_gene_ids)]:
        p = archive/f'generated/models/{field}_{key}_PNA.pt'
        # Trusted local training artifacts contain NumPy scalar lists, not only tensors.
        raw = torch.load(p, map_location='cpu', weights_only=False)
        # Training tensors use 0..4; exported NIAPU-style labels use 1..5.
        raw_values = np.asarray(raw).ravel() + (0 if field == 'test_ids' else 1)
        np.testing.assert_array_equal(raw_values, np.asarray(values))
        paths[field] = p
    matrices = [blocks[0], np.column_stack(blocks), blocks[1]]
    out = base/'accumulated_results/niapu_comparison'
    for folder in ['data','plots','cache']:(out/folder).mkdir(parents=True,exist_ok=True)
    # Match plot_gnn.plot_features using this archived run's own configuration.
    settings = dict(n_neighbors=int(saved.umap_n_neighbors),
                    min_dist=float(saved.umap_min_dist), metric=str(saved.umap_metric),
                    random_state=int(saved.SEED), n_jobs=1, n_components=2)
    provenance = dict(run=run_name, source_sha256={k:digest(v) for k,v in paths.items()},
                      feature_blocks=info, settings=settings, umap_version=__import__('importlib.metadata',fromlist=['version']).version('umap-learn'))
    token = hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()
    cache = out/'cache'/f'niapu_umap_{run_name}.npz';stamp=cache.with_suffix('.json')
    if cache.exists() and stamp.exists() and json.loads(stamp.read_text()).get('token') == token:
        with np.load(cache) as z:embeddings=[z[f'embedding_{i}'] for i in range(3)]
        print(f'{disease}: reuse UMAP coordinates from {run_name}',flush=True)
    else:
        from umap import UMAP
        embeddings=[]
        for label,x in zip(['NeDBIT','NeDBIT + DNA methylation','DNA methylation'], matrices):
            print(f'{disease}: UMAP {label}: {x.shape}',flush=True)
            embeddings.append(UMAP(**settings).fit_transform(x))  # Never pass labels to UMAP.
        np.savez_compressed(cache, **{f'embedding_{i}':e for i,e in enumerate(embeddings)})
        stamp.write_text(json.dumps(dict(token=token,**provenance),indent=2))
    # The saved second forward output is h after the final residual PNAConv
    # and LayerNorm, not the concatenated input/states or class probabilities.
    latent_path = archive/f'generated/models/embed_conv_{key}_PNA.pt'
    latent = np.asarray(torch.load(latent_path, map_location='cpu', weights_only=False), dtype=float)
    if latent.ndim != 2 or latent.shape[0] != len(names) or not np.isfinite(latent).all():
        raise ValueError('Final PNA representation must match held-out prediction rows and be finite')
    latent_provenance = dict(layer='Final PNA message-passing output after residual LayerNorm (forward output h)',
                             source=str(latent_path), source_sha256=digest(latent_path),
                             shape=list(latent.shape), scaling='Saved latent values, without additional scaling',
                             checkpoint_sha256=provenance['source_sha256']['checkpoint'],
                             node_ids_sha256=provenance['source_sha256']['test_ids'],
                             settings=settings, umap_version=provenance['umap_version'])
    latent_token = hashlib.sha256(json.dumps(latent_provenance,sort_keys=True).encode()).hexdigest()
    latent_cache = out/'cache'/f'niapu_umap_pna_last_layer_{run_name}.npz'
    latent_stamp = latent_cache.with_suffix('.json')
    if latent_cache.exists() and latent_stamp.exists() and json.loads(latent_stamp.read_text()).get('token') == latent_token:
        with np.load(latent_cache) as z:latent_embedding=z['embedding']
        print(f'{disease}: reuse final-layer PNA UMAP from {run_name}',flush=True)
    else:
        from umap import UMAP
        print(f'{disease}: UMAP final PNA layer: {latent.shape}',flush=True)
        latent_embedding=UMAP(**settings).fit_transform(latent)
        np.savez_compressed(latent_cache,embedding=latent_embedding)
        latent_stamp.write_text(json.dumps(dict(token=latent_token,**latent_provenance),indent=2))
    matrices.append(latent);embeddings.append(latent_embedding)
    trusts = [float(trustworthiness(x,e,n_neighbors=15,metric=settings['metric'])) for x,e in zip(matrices,embeddings)]
    records=[];coordinates=[]
    for i,(space,label_source,j) in enumerate([('NeDBIT','NIAPU',0),('NeDBIT + DNA methylation','NIAPU',1),
                                              ('DNA methylation','NIAPU',2),('PNA final-layer representation','PNA',3)]):
        labels=pred if label_source=='PNA' else truth
        records.append(dict(panel='ABCD'[i], features=space, labels=label_source,n_nodes=len(names),
                            n_classes=len(np.unique(labels)), **quality(matrices[j],embeddings[j],labels),
                            trustworthiness=trusts[j],PNA_NIAPU_ARI=float(adjusted_rand_score(truth,pred))))
        coordinates.append(pd.DataFrame(dict(panel='ABCD'[i],node_name=names,graph_node_id=predictions.test_gene_ids,
            niapu_class=truth,pna_class=pred,display_class=labels,UMAP1=embeddings[j][:,0],UMAP2=embeddings[j][:,1])))
    pd.DataFrame(records).to_csv(out/'data/niapu_umap_quality.tsv',sep='\t',index=False)
    pd.concat(coordinates,ignore_index=True).to_csv(out/'data/niapu_umap_coordinates.tsv',sep='\t',index=False)
    manifest=dict(**provenance,pna_representation=latent_provenance,n_nodes=len(names),n_unique_CpGs=len(set(n.split('_')[0] for n in names)),
                  niapu_counts={int(k):int(v) for k,v in zip(*np.unique(truth,return_counts=True))},
                  pna_counts={int(k):int(v) for k,v in zip(*np.unique(pred,return_counts=True))},
                  status='complete',display_label_source={'A':'NIAPU','B':'NIAPU','C':'NIAPU','D':'PNA predicted'},
                  scores_displayed=False,subtitle_displayed=False,predictions='Archived held-out inference after restoring the best PNA checkpoint; raw tensors checked against table',
                  scaling='Median imputation and z scores on the plotted nodes; each block divided by sqrt(nonconstant column count); equal total block variance',
                  supervised_umap=False,raw_class_tensor_to_NIAPU_offset=1,
                  display_palette=dict(name='tab10',alpha=.9,classes={k:dict(label=v[0],color=v[1]) for k,v in CLASSES.items()}))
    render(out,disease,run_name,embeddings,truth,pred,records)
    (out/'umap_manifest.json').write_text(json.dumps(manifest,indent=2))
    findings = 'All held-out test nodes are included without a likely-positive filter. Panels A–C are colored by NIAPU labels and D by PNA predicted labels; scores and the subtitle are omitted from the figures.'
    (out/'UMAP_REPORT.md').write_text(f'''# {disease}: NIAPU/PNA UMAP comparison

{findings} Scores remain available in the quality TSV. Negative silhouette values indicate that nodes are, on average, closer to another class than their own; values near zero indicate overlap. These observations concern this representation and node set, not general disease discrimination.

Four matched panels use {len(names):,} held-out CpG–gene nodes ({manifest['n_unique_CpGs']:,} unique CpGs) from archived PNA {run_name}. These are all five NIAPU classes, not the externally selected likely-positive-only classifier panel. NIAPU labels are checked against both archived feature labels and current rankings. PNA predictions are the archived best-checkpoint test predictions, independently checked against raw saved prediction, true-label and node-ID tensors. Model checkpoint and input hashes are in umap_manifest.json. No model is retrained and no external cohort is downloaded for UMAP.

Colors match the existing training UMAPs (tab10): positive seed blue, likely positive orange, weakly negative green, likely negative red, and reliable negative purple; opacity is 0.9.

A: NeDBIT features colored by NIAPU labels. B: NeDBIT + DNA methylation colored by NIAPU labels. C: DNA methylation colored by NIAPU labels. D: a separate UMAP of the {latent.shape[1]}-dimensional final PNA message-passing representation, colored by PNA predictions. The archived forward output h is the final residual PNAConv output after LayerNorm, saved as embed_conv; it is not the concatenated input/states, class logits or probabilities. It uses the same UMAP settings as A–C and the same held-out node order, verified against saved node IDs. Saved latent values enter UMAP without additional scaling. DNA methylation columns are patient measurements describing CpG nodes, not patient embeddings. Every panel uses the same held-out nodes. Multiple gene annotations can represent the same CpG and are not independent observations.

Each feature column is median-imputed if needed and standardized on the plotted nodes; constant columns are removed. Each block is divided by the square root of its nonconstant feature count so the many methylation measurements do not overwhelm the NeDBIT block by column count. This is exploratory visualization scaling, not a new PNA input transformation. The archived model's original predictions and final-layer representation are used unchanged. UMAP is unsupervised: labels never enter fitting. UMAP parameters come directly from the archived run configuration, matching the earlier training UMAPs: seed={settings['random_state']}, metric={settings['metric']}, neighbors={settings['n_neighbors']}, min_dist={settings['min_dist']}; all settings are shared across feature spaces. Coordinates are cached with input/checkpoint hashes and UMAP version.

Scores saved to the TSV, without display on figures: silhouette in 2D (higher), silhouette in standardized/block-balanced feature space for A–C or saved PNA latent space for D (higher), Davies–Bouldin in 2D (lower), and trustworthiness at 15 neighbors (higher). The TSV additionally includes Calinski–Harabasz in 2D (higher), node/class totals and NIAPU/PNA adjusted Rand agreement. These metrics measure separation of the supplied labels; no unsupervised clustering or disease classification is performed. Silhouette and Davies–Bouldin scores use Euclidean distances; trustworthiness uses the configured UMAP input metric. Trustworthiness measures preservation of feature-space neighborhoods and does not depend on colors, and is calculated separately for each input representation. Feature-space silhouette is the stronger check on apparent UMAP separation: projection scores alone can be misleading. NIAPU derives labels from graph propagation and NeDBIT describes that graph, so their separation is partly expected. PNA predicts NIAPU classes; sharper predicted-label separation is not independent biological validation. Scores are descriptive, with no independence-based P values.

See data/niapu_umap_quality.tsv, data/niapu_umap_coordinates.tsv and plots/niapu_umap_comparison.pdf (also SVG/300-dpi PNG and individual panels). Rerunning a different --run replaces the displayed plots/tables; per-run feature and final-layer coordinate caches remain available.
''')
    print(pd.DataFrame(records).to_string(index=False),flush=True)
    print(f'{disease}: UMAP complete -> {out}',flush=True)
    return out
