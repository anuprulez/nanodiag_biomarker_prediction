"""Candidate-only four-model comparison of original CV and external transfer."""
from pathlib import Path
import json
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score, confusion_matrix
from .publication import STYLE, save_publication

COHORTS = [('original_patient_CV', 'Original cohort', '#0072B2'),
           ('external_transfer_selected_cohort', 'External cohort', '#D55E00')]
MODELS = [('Random Forest', 'Random Forest'),
          ('XGBoost', 'XGBoost'), ('TabPFN3', 'TabPFN'), ('TabICLv2', 'TabICL')]

def render_candidate_comparison(out, disease, comparison_label=None):
    out = Path(out)
    predictions = pd.read_csv(out/'data/classifier_patient_predictions.tsv', sep='\t')
    predictions = predictions.loc[predictions.panel.eq('candidates_only') & predictions.model.isin([m for m, _ in MODELS])].copy()
    metrics = pd.read_csv(out/'data/classifier_performance.tsv', sep='\t')
    metrics = metrics.loc[metrics.panel.eq('candidates_only') & metrics.model.isin([m for m, _ in MODELS])].copy()
    n = int(metrics.n_features.iloc[0])
    assert metrics.n_features.eq(n).all()
    assert set(predictions.model) == {m for m, _ in MODELS}
    archive = (out/'retired_classifier_comparisons' if comparison_label
               else out.parent/'retired_classifier_comparisons'/disease)
    archive.mkdir(parents=True, exist_ok=True)
    for name in ('unweighted', 'inverse_class_weighted', 'classifier_plot_sets.json'):
        source = out/name
        if source.exists():
            target = archive/name
            if target.exists():
                target = archive/(name+'.previous')
            if target.exists():
                raise FileExistsError(target)
            shutil.move(str(source), str(target))
    for source in list((out/'plots').glob('*')):
        if ('classifier' in source.name or source.stem in ('external_transfer_selected_cohort_performance', 'original_patient_CV_performance')) and not source.name.startswith('candidate_cohort_'):
            target = archive/'plots'/source.name
            target.parent.mkdir(exist_ok=True)
            if target.exists(): target.unlink()
            shutil.move(str(source), str(target))
    predictions.to_csv(out/'data/candidate_cohort_predictions.tsv', sep='\t', index=False)
    metrics.to_csv(out/'data/candidate_cohort_performance.tsv', sep='\t', index=False)
    title = f'{"Breast cancer" if disease == "BC" else disease}: {n} candidate CpGs'
    if comparison_label:title = title.replace(': ',f': {comparison_label} | ',1)
    figures = []
    with plt.rc_context(STYLE):
        for kind in ('ROC', 'PRC'):
            fig, axes = plt.subplots(2, 2, figsize=(12, 11))
            for ax, (model, label) in zip(axes.flat, MODELS):
                for cohort, cohort_label, color in COHORTS:
                    frame = predictions.loc[predictions.model.eq(model) & predictions.cohort.eq(cohort)]
                    y, p = frame.label.to_numpy(), frame.probability.to_numpy()
                    if kind == 'ROC':
                        x, z, _ = roc_curve(y, p)
                        ax.plot(x, z, color=color, lw=2.4, label=f'{cohort_label}: AUROC {roc_auc_score(y,p):.3f}')
                    else:
                        z, x, _ = precision_recall_curve(y, p)
                        # Ascending recall preserves all threshold points, including the vertical
                        # segment at recall=1 down to the observed class prevalence.
                        ax.step(x[::-1], z[::-1], where='pre', color=color, lw=2.8,
                                label=f'{cohort_label}: AP {average_precision_score(y,p):.3f}')
                        ax.plot([1], [y.mean()], marker='o', color=color, ms=6, clip_on=False)
                        ax.axhline(y.mean(), color=color, ls=':', lw=1.1, alpha=.65)
                if kind == 'ROC': ax.plot([0,1],[0,1], ':', color='#aaaaaa', lw=1)
                ax.set(xlim=(-.02,1.02), ylim=(-.02,1.02), title=label,
                       xlabel='1 − specificity' if kind == 'ROC' else 'Recall',
                       ylabel='Sensitivity' if kind == 'ROC' else 'Precision')
                ax.set_xticks(np.linspace(0,1,6)); ax.set_yticks(np.linspace(0,1,6))
                ax.grid(color='#e8e8e8', lw=.6); ax.set_axisbelow(True)
                ax.spines[['top','right']].set_visible(False)
                ax.tick_params(labelsize=15)
                ax.set_xlabel(ax.get_xlabel(), fontsize=18); ax.set_ylabel(ax.get_ylabel(), fontsize=18)
                ax.set_title(label, fontsize=20, pad=12)
                ax.legend(loc='lower right' if kind == 'ROC' else 'lower left', fontsize=12)
            fig.suptitle(title + (' | Precision–recall curve' if kind == 'PRC' else ' | ROC curve'), fontsize=23)
            fig.tight_layout(rect=(0,0,1,.96), h_pad=2, w_pad=2)
            figures.append((f'candidate_cohort_{kind}', fig))
        # Original layout: cohort panels A/B in rows, four models in columns.
        fig, axes = plt.subplots(2, len(MODELS), figsize=(17, 10.5))
        fig.subplots_adjust(left=.085, right=.90, bottom=.12, top=.83,
                            wspace=.90, hspace=.95)
        for row, (cohort, cohort_label, color) in enumerate(COHORTS):
            cmap = LinearSegmentedColormap.from_list(cohort, ['#ffffff', color])
            for col, (model, label) in enumerate(MODELS):
                frame = predictions.loc[predictions.model.eq(model) & predictions.cohort.eq(cohort)]
                matrix = confusion_matrix(frame.label, frame.predicted_label_at_0_5, labels=[0,1])
                fractions = matrix / matrix.sum(axis=1, keepdims=True)
                ax = axes[row,col]
                image = ax.imshow(fractions, cmap=cmap, vmin=0, vmax=1)
                for i in range(2):
                    for j in range(2):
                        ax.text(j,i,f'{matrix[i,j]:,}\n({fractions[i,j]:.1%})',
                                ha='center',va='center',fontsize=15.5,
                                color='white' if fractions[i,j]>.6 else '#202020')
                labels = ['Healthy', 'Breast cancer' if disease=='BC' else 'AML']
                ax.set_xticks([0,1],labels); ax.set_yticks([0,1],labels)
                ax.set_xlabel('Predicted class', fontsize=17, labelpad=8)
                ax.set_ylabel('Ground truth', fontsize=17, labelpad=8)
                ax.set_title(label, fontsize=20, pad=13)
                ax.tick_params(length=0,labelsize=14,pad=6)
                ax.tick_params(axis='x', labelrotation=25)
                for spine in ax.spines.values(): spine.set_visible(False)
            fig.canvas.draw()
            box = axes[row,-1].get_position()
            heading = ('A  Original cohort: five-fold cross-validation' if row == 0 else
                       'B  Train on original cohort, predict on external cohort')
            fig.text(.50, box.y1+.075, heading,ha='center',fontsize=20)
            color_axis = fig.add_axes([.935, box.y0, .012, box.height])
            colorbar = fig.colorbar(image, cax=color_axis, ticks=[0,.5,1])
            colorbar.ax.set_yticklabels(['0%', '50%', '100%'])
            colorbar.set_label('Within-class percentage', fontsize=15, labelpad=10)
            colorbar.ax.tick_params(labelsize=13, length=3)
            colorbar.outline.set_visible(False)
        fig.suptitle(title, fontsize=25,y=.995)
        figures.append(('candidate_cohort_confusion_matrices',fig))
        with PdfPages(out/'candidate_cohort_comparison.pdf') as pdf:
            for stem, fig in figures:
                pdf.savefig(fig,bbox_inches='tight',facecolor='white')
                save_publication(fig,out,stem)
    (out/'candidate_cohort_comparison.json').write_text(json.dumps(dict(
        disease=disease,n_features=n,feature_panel='candidates_only',
        source_predictions='data/classifier_patient_predictions.tsv',
        models=[label for _,label in MODELS],class_weighting=False,cohort_colors={label:color for _,label,color in COHORTS},
        original_evaluation='five-fold out-of-fold patient predictions',
        external_evaluation='fit on all original patients; predict external patients',
        retired_outputs=str(archive)),indent=2)+'\n')
    text=f'''# {title}: original versus external cohort

[Complete figures](candidate_cohort_comparison.pdf): ROC, precision–recall and confusion matrices.

Only the fixed {n}-CpG candidate panel is displayed. Four models: Random Forest, XGBoost, TabPFN and TabICL. Logistic regression is excluded from the displayed comparison. Original-cohort results use five-fold patient cross-validation; external results use models fitted on all original patients. No class weighting or probability-prior correction is applied. Original cohort is blue (#0072B2); external cohort is vermilion (#D55E00), matching the delta-beta forest plot.

Precision–recall curves include all score thresholds and mark the recall=1 endpoint, where precision equals the disease prevalence. Dotted lines show cohort-specific prevalence. Curves are not extended to an artificial zero precision.

Saved predictions are reused without retraining. Seed panels and separate weighting variants are excluded from these comparison figures. Retired comparison outputs are preserved outside this results folder in `{archive}`.

Candidate selection used external statistics; these are exploratory cohort comparisons, not untouched validation. Study and disease status are confounded in the external cohort.
'''
    (out/'CLASSIFIER_COMPARISON.md').write_text(text)
    report=out/'REPORT.md'
    if report.exists():
        content=report.read_text()
        cut=content.find('\n## Foundation classifier weighting')
        if cut>=0:content=content[:cut]
        cut=content.find('\n## Current classifier comparison')
        if cut>=0:content=content[:cut]
        content+='\n## Current classifier comparison\n\n'+text.split('\n',1)[1]
        report.write_text(content)
    print(f'{disease}: rendered candidate-only cohort comparison ({n} CpGs)',flush=True)
