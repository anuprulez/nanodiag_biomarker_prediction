#!/usr/bin/env python3
"""Replot frozen GSE52865 predictions without refitting or threshold selection."""
from pathlib import Path
import hashlib
import json
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_curve, precision_recall_curve, roc_auc_score,
                             average_precision_score, confusion_matrix, f1_score,
                             accuracy_score, balanced_accuracy_score)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from post_evaluation_support.publication import STYLE, save_publication
MODELS = {'Random Forest': 'RF', 'TabPFN3': 'TabPFN', 'TabICLv2': 'TabICL', 'XGBoost': 'XGBoost'}
COLORS = ['#7B2CBF', '#009E73', '#CC79A7', '#D55E00']

def main():
    source = ROOT/'data_corr_0.33_knn_n_3000_n_neg_10000/accumulated_results/post_evaluation/independent_feature_verification'
    out = ROOT/'evaluation/independent_classification/GSE52865'
    out.mkdir(parents=True, exist_ok=True)
    path = source/'data/patient_predictions.tsv'
    predictions = pd.read_csv(path, sep='\t')
    assert set(predictions.series) == {'GSE52865'}
    assert set(predictions.panel) == {'candidates_only'}
    assert set(predictions.model) == set(MODELS)
    assert not predictions.duplicated(['sample_id', 'model']).any()
    rows = []
    for name in MODELS:
        d = predictions.loc[predictions.model.eq(name)]
        y, p = d.label.to_numpy(int), d.probability.to_numpy(float)
        assert len(d) == 57 and (y == 1).sum() == 40 and (y == 0).sum() == 17
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        predicted = (p >= .5).astype(int)
        assert np.array_equal(predicted, d.predicted_label_at_0_5)
        tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
        rows.append(dict(model=name, series='GSE52865', n_CpGs=33, n_samples=57,
                         macro_F1=f1_score(y, predicted, average='macro'),
                         accuracy=accuracy_score(y, predicted),
                         balanced_accuracy=balanced_accuracy_score(y, predicted),
                         AUROC=roc_auc_score(y, p), AP=average_precision_score(y, p),
                         TN=tn, FP=fp, FN=fn, TP=tp, threshold=.5))
    metrics = pd.DataFrame(rows)
    old = pd.read_csv(source/'data/classifier_metrics.tsv', sep='\t').set_index('model')
    for row in metrics.itertuples():
        assert all(getattr(row, key) == old.loc[row.model, key] for key in ['TN', 'FP', 'FN', 'TP'])
    predictions.to_csv(out/'patient_predictions.tsv', sep='\t', index=False)
    metrics.to_csv(out/'classification_metrics.tsv', sep='\t', index=False)
    title = 'Breast cancer: independent GSE52865 | 33 fixed candidate CpGs'
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), layout='constrained')
        for (name, display), color in zip(MODELS.items(), COLORS):
            d = predictions.loc[predictions.model.eq(name)]
            y, p = d.label.to_numpy(int), d.probability.to_numpy(float)
            fpr, tpr, _ = roc_curve(y, p)
            precision, recall, _ = precision_recall_curve(y, p)
            axes[0].plot(fpr, tpr, color=color, lw=2.5, label=f'{display} · AUROC = {roc_auc_score(y,p):.3f}')
            axes[1].step(recall, precision, where='post', color=color, lw=2.5, label=f'{display} · AP = {average_precision_score(y,p):.3f}')
        axes[0].plot([0,1], [0,1], ':', color='#999999')
        axes[1].axhline(40/57, ls=':', color='#999999', label='Cancer prevalence = 0.702')
        for i, ax in enumerate(axes):
            ax.set(xlim=(-.025,1.025), ylim=(-.025,1.025), xlabel='1 − specificity' if i == 0 else 'Recall',
                   ylabel='Sensitivity' if i == 0 else 'Precision', title='A. ROC' if i == 0 else 'B. Precision–recall')
            ax.set_aspect('equal'); ax.grid(color='#e8e8e8', lw=.6)
            ax.spines[['top','right']].set_visible(False)
            ax.legend(loc='lower right' if i == 0 else 'lower left', fontsize=10)
        fig.suptitle(title+'\nOriginal-trained models · no class weighting', fontsize=17)
        save_publication(fig, out, 'BC_33_CpGs_GSE52865_ROC_PR')
        fig, axes = plt.subplots(1,4,figsize=(17,5), layout='constrained')
        for ax, row in zip(axes, metrics.itertuples()):
            cm = np.array([[row.TN,row.FP],[row.FN,row.TP]])
            ax.imshow(cm, cmap='Blues', vmin=0, vmax=40)
            for i in range(2):
                for j in range(2):
                    ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=20,color='white' if cm[i,j]>20 else '#202020')
            ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Normal','Cancer'],yticklabels=['Normal','Cancer'],
                   xlabel='Predicted',ylabel='Observed',title=MODELS[row.model])
        fig.suptitle(title+'\nDecision threshold = 0.5',fontsize=17)
        save_publication(fig,out,'BC_33_CpGs_GSE52865_confusion_matrices')
        fig, ax = plt.subplots(figsize=(8,3.5),layout='constrained')
        values = metrics.macro_F1.to_numpy()[None,:]
        im = ax.imshow(values,cmap='YlGnBu',vmin=0,vmax=1,aspect='auto')
        ax.set_xticks(range(4),list(MODELS.values()));ax.set_yticks([0],['GSE52865\n57 specimens'])
        for j,v in enumerate(values[0]):ax.text(j,0,f'{v:.3f}',ha='center',va='center',color='white',fontsize=16)
        ax.set_title('Independent test macro-F1 · 33 BC CpGs',pad=15)
        fig.colorbar(im,ax=ax,label='Macro-F1',ticks=np.linspace(0,1,6))
        save_publication(fig,out,'BC_33_CpGs_GSE52865_macro_F1')
    (out/'replot_manifest.json').write_text(json.dumps(dict(dataset='GSE52865',n_CpGs=33,
        source_predictions=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        procedure='Saved original-trained probabilities; fixed threshold 0.5; no refitting, feature selection or tuning',
        source_report=str(source/'REPORT.md')),indent=2)+'\n')
    (out/'README.md').write_text('# Independent BC disease classification\n\nGSE52865: 40 breast tumours and 17 morphologically normal breast specimens; 33 fixed candidate CpGs. Replotted from frozen original-trained predictions, with no refitting or threshold optimization. All confusion counts reproduce the previous evaluation. Macro-F1 is the equally weighted mean of normal-class and cancer-class F1 at threshold 0.5.\n\nThis is an independent test, so no validation scores are invented. See replot_manifest.json for the source and its hash. The original report documents possible shared donors and the limits of the small tissue cohort.\n')
    print(metrics.to_string(index=False))
    print(out)

if __name__ == '__main__':
    main()
