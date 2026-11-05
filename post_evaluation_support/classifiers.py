"""Patient-level logistic regression, random forest and XGBoost comparisons."""
from pathlib import Path
import argparse
import json
import sys
from itertools import combinations
from importlib.metadata import version

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (roc_curve, precision_recall_curve, confusion_matrix,
                             accuracy_score, balanced_accuracy_score, f1_score,
                             roc_auc_score, average_precision_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

from .analysis import (RANDOM_SEED, CAUTION, estimator, score, bootstrap_metrics,
                       original_metadata, digest)
from .publication import STYLE, save_publication

MODELS = ('Logistic regression', 'Random Forest', 'XGBoost')
FOUNDATION_MODELS=('TabPFN3','TabICLv2')
MODEL_ORDER=MODELS+FOUNDATION_MODELS
COLORS=dict(zip(MODEL_ORDER,('#0072B2','#7B2CBF','#D55E00','#009E73','#CC79A7')))

def plotted_models(predictions):
    return tuple(name for name in MODEL_ORDER if name in set(predictions.model))

def weighting_title(predictions,weighted):
    if not weighted:return 'No class weighting'
    return 'Inverse weighting / foundation prior correction' if set(predictions.model)&set(FOUNDATION_MODELS) else 'Inverse-frequency class weighting'

PARAMETERS = {
    'Logistic regression': dict(C=1., solver='liblinear', max_iter=2000, random_state=RANDOM_SEED),
    'Random Forest': dict(n_estimators=500, min_samples_leaf=2, max_features='sqrt',
                          class_weight=None, n_jobs=4, random_state=RANDOM_SEED),
    'XGBoost': dict(n_estimators=500, max_depth=3, learning_rate=.03, subsample=.8,
                    colsample_bytree=.8, reg_lambda=1., scale_pos_weight=1.,
                    objective='binary:logistic', eval_metric='logloss', tree_method='hist',
                    n_jobs=4, random_state=RANDOM_SEED),
}


def patient_estimator(name):
    if name == 'Logistic regression':
        return estimator()
    model = RandomForestClassifier(**PARAMETERS[name]) if name == 'Random Forest' else XGBClassifier(**PARAMETERS[name])
    return make_pipeline(SimpleImputer(strategy='median'), model)


def balanced_training_weights(labels):
    """Normalize inverse frequency to mean weight 1; use training labels only."""
    labels=np.asarray(labels,dtype=int)
    counts=np.bincount(labels,minlength=2)
    if set(np.unique(labels))!={0,1}:
        raise ValueError('Inverse weighting requires both training classes 0 and 1')
    weights=len(labels)/(2*counts.astype(float))
    return weights[labels],counts,weights


def fit_patient_model(name, features, labels, inverse_class_weighting=False):
    model=patient_estimator(name)
    kwargs={}
    if inverse_class_weighting:
        weights,_,_=balanced_training_weights(labels)
        kwargs[model.steps[-1][0]+'__sample_weight']=weights
    return model.fit(features,labels,**kwargs)


def classification_metrics(y, p):
    predicted = (p >= .5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return dict(accuracy_at_0_5=accuracy_score(y, predicted),
                balanced_accuracy_at_0_5=balanced_accuracy_score(y, predicted),
                F1_at_0_5=f1_score(y, predicted, zero_division=0),
                specificity_at_0_5=tn/(tn+fp), sensitivity_at_0_5=tp/(tp+fn),
                TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp))


def comparison_figure(predictions, metrics, disease, cohort, inverse_class_weighting=False):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(14.4, 12.5))
        for row, panel in enumerate(('seeds_only', 'candidates_only')):
            selected = predictions.loc[(predictions.panel == panel) & (predictions.cohort == cohort)]
            n_features = int(metrics.loc[(metrics.panel == panel) & (metrics.cohort == cohort), 'n_features'].iloc[0])
            for name in plotted_models(predictions):
                data = selected.loc[selected.model == name]
                y, p = data.label.to_numpy(int), data.probability.to_numpy(float)
                fpr, tpr, _ = roc_curve(y, p)
                precision, recall, _ = precision_recall_curve(y, p)
                axes[row, 0].plot(fpr, tpr, color=COLORS[name], lw=2.6,
                                  label=f'{name} · AUROC = {roc_auc_score(y,p):.3f}')
                axes[row, 1].step(recall, precision, where='post', color=COLORS[name], lw=2.6,
                                  label=f'{name} · AP = {average_precision_score(y,p):.3f}')
            title = 'Seeds only' if panel == 'seeds_only' else 'Candidates only'
            axes[row, 0].plot([0, 1], [0, 1], ':', color='#aaaaaa', lw=1)
            axes[row, 1].axhline(y.mean(), ls=':', color='#aaaaaa', lw=1)
            for column, ax in enumerate(axes[row]):
                ax.set_title(f'{chr(65+row*2+column)}   {title} ({n_features} CpGs)', fontsize=17, pad=15)
                ax.set(xlim=(-.025,1.025), ylim=(-.025,1.025),
                       xlabel='1 − specificity' if column == 0 else 'Recall',
                       ylabel='Sensitivity' if column == 0 else 'Precision')
                ax.set_xticks(np.linspace(0,1,6));ax.set_yticks(np.linspace(0,1,6))
                ax.set_aspect('equal');ax.set_axisbelow(True)
                ax.grid(color='#e8e8e8',lw=.65)
                ax.spines[['top','right']].set_visible(False)
                ax.legend(loc='lower right' if column == 0 else 'lower left',fontsize=10.5,
                          frameon=True,facecolor='white',edgecolor='#dddddd',framealpha=.95)
        name='Breast cancer' if disease == 'BC' else disease
        context='Original cohort · five-fold patient CV' if cohort == 'original_patient_CV' else 'External cohort · original-trained models'
        title=f'{name} | {context}'
        title+='\n'+weighting_title(predictions,inverse_class_weighting)
        fig.suptitle(title,fontsize=18,y=.992)
        fig.subplots_adjust(left=.08,right=.985,bottom=.07,top=.89,hspace=.36,wspace=.25)
    return fig


def single_curve_figure(predictions, metrics, disease, cohort, kind, inverse_class_weighting=False):
    with plt.rc_context(STYLE):
        fig,axes=plt.subplots(1,2,figsize=(14.4,7))
        for ax,panel in zip(axes,('seeds_only','candidates_only')):
            subset=predictions.loc[(predictions.panel==panel)&(predictions.cohort==cohort)]
            n=int(metrics.loc[(metrics.panel==panel)&(metrics.cohort==cohort),'n_features'].iloc[0])
            for model in plotted_models(predictions):
                data=subset.loc[subset.model==model]
                y=data.label.to_numpy(int);p=data.probability.to_numpy(float)
                if kind=='ROC':
                    x,z,_=roc_curve(y,p)
                    ax.plot(x,z,color=COLORS[model],lw=2.7,label=f'{model} · AUROC = {roc_auc_score(y,p):.3f}')
                else:
                    z,x,_=precision_recall_curve(y,p)
                    ax.step(x,z,where='post',color=COLORS[model],lw=2.7,label=f'{model} · AP = {average_precision_score(y,p):.3f}')
            if kind=='ROC':ax.plot([0,1],[0,1],':',color='#aaaaaa',lw=1)
            else:ax.axhline(y.mean(),ls=':',color='#aaaaaa',lw=1)
            ax.set(xlim=(-.025,1.025),ylim=(-.025,1.025),
                   xlabel='1 − specificity' if kind=='ROC' else 'Recall',
                   ylabel='Sensitivity' if kind=='ROC' else 'Precision',
                   title=f'{"Seeds only" if panel=="seeds_only" else "Candidates only"} ({n} CpGs)')
            ax.set_xticks(np.linspace(0,1,6));ax.set_yticks(np.linspace(0,1,6));ax.set_aspect('equal')
            ax.grid(color='#e8e8e8',lw=.65);ax.set_axisbelow(True)
            ax.spines[['top','right']].set_visible(False)
            ax.legend(loc='lower right' if kind=='ROC' else 'lower left',fontsize=10.5,
                      facecolor='white',edgecolor='#dddddd',framealpha=.95)
        context='External cohort' if cohort=='external_transfer_selected_cohort' else 'Original-patient CV'
        title=f'{"Breast cancer" if disease=="BC" else disease} | {context} | {"ROC" if kind=="ROC" else "Precision–recall"}'
        title+='\n'+weighting_title(predictions,inverse_class_weighting)
        fig.suptitle(title,fontsize=19,y=.99)
        fig.subplots_adjust(left=.07,right=.985,bottom=.11,top=.80,wspace=.25)
    return fig


def breast_cancer_unweighted_confusion_figure(predictions, feature_counts):
    """Unweighted original CV and external predictions, grouped by cohort."""
    models = plotted_models(predictions)
    display_names = {'TabPFN3': 'TabPFN', 'TabICLv2': 'TabICL'}
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(18, 17))
        grid = fig.add_gridspec(10, len(models),
            height_ratios=[.45, .30, 3, .30, 3, .65, .30, 3, .30, 3],
            left=.065, right=.985, bottom=.045, top=.98, wspace=.65, hspace=.50)
        cohorts = [('original_patient_CV', 'A', 'Breast cancer: original cohort cross-validation'),
                   ('external_transfer_selected_cohort', 'B', 'Breast cancer: train on original cohort, predict on external')]
        for group, (cohort, letter, title) in enumerate(cohorts):
            offset = group * 5
            heading = fig.add_subplot(grid[offset, :]); heading.set_axis_off()
            heading.text(0, .5, letter, fontsize=23, fontweight='bold', va='center')
            heading.text(.5, .5, title, fontsize=18, ha='center', va='center')
            for row, panel in enumerate(('seeds_only', 'candidates_only')):
                n = int(feature_counts[panel])
                caption = fig.add_subplot(grid[offset + 1 + 2*row, :]); caption.set_axis_off()
                caption.text(.5, .5, f'{"Seeds" if row == 0 else "Candidates"}: {n} CpGs',
                             fontsize=16, ha='center', va='center')
                for column, name in enumerate(models):
                    data = predictions.loc[(predictions.panel == panel) &
                                           (predictions.model == name) &
                                           (predictions.cohort == cohort)]
                    matrix = confusion_matrix(data.label, data.predicted_label_at_0_5, labels=[0, 1])
                    fractions = matrix / matrix.sum(axis=1, keepdims=True)
                    ax = fig.add_subplot(grid[offset + 2 + 2*row, column])
                    ax.imshow(fractions, cmap='Blues', vmin=0, vmax=1)
                    for i in range(2):
                        for j in range(2):
                            ax.text(j, i, f'{matrix[i,j]:,}\n({fractions[i,j]:.1%})',
                                    ha='center', va='center', fontsize=14,
                                    color='white' if fractions[i,j] > .55 else '#202020')
                    ax.set_xticks([0, 1], ['Healthy', 'Cancer'])
                    ax.set_yticks([0, 1], ['Healthy', 'Cancer'])
                    ax.set_xlabel('Predicted class', fontsize=13, labelpad=7)
                    ax.set_ylabel('Ground truth', fontsize=13, labelpad=7)
                    ax.set_title(display_names.get(name, name), fontsize=15, pad=10)
                    ax.tick_params(length=0, labelsize=12, pad=6)
                    for spine in ax.spines.values():
                        spine.set_visible(False)
        return fig


def confusion_figure(predictions, disease, cohort='external_transfer_selected_cohort', inverse_class_weighting=False, feature_counts=None):
    if disease == 'BC' and cohort == 'external_transfer_selected_cohort' and not inverse_class_weighting and feature_counts is not None:
        return breast_cancer_unweighted_confusion_figure(predictions, feature_counts)
    with plt.rc_context(STYLE):
        fig,axes=plt.subplots(2,len(plotted_models(predictions)),figsize=(4.5*len(plotted_models(predictions)),9))
        for row,panel in enumerate(('seeds_only','candidates_only')):
            for column,name in enumerate(plotted_models(predictions)):
                data=predictions.loc[(predictions.panel==panel)&(predictions.model==name)&
                                     (predictions.cohort==cohort)]
                matrix=confusion_matrix(data.label,data.predicted_label_at_0_5,labels=[0,1])
                fractions=matrix/matrix.sum(axis=1,keepdims=True)
                ax=axes[row,column];ax.imshow(fractions,cmap='Blues',vmin=0,vmax=1)
                for i in range(2):
                    for j in range(2):
                        ax.text(j,i,f'{matrix[i,j]:,}\n({fractions[i,j]:.1%})',ha='center',va='center',
                                color='white' if fractions[i,j]>.55 else '#202020',fontsize=15)
                ax.set_xticks([0,1],['Healthy',disease]);ax.set_yticks([0,1],['Healthy',disease])
                ax.set_xlabel('Predicted patient class');ax.set_ylabel('Observed patient class')
                ax.set_title(('Seeds only' if row==0 else 'Candidates only')+'\n'+name,fontsize=15,pad=12)
                ax.tick_params(length=0)
        context='External patients' if cohort=='external_transfer_selected_cohort' else 'Original-patient CV'
        title=f'{"Breast cancer" if disease=="BC" else disease} | {context} (threshold = 0.50)'
        title+='\n'+weighting_title(predictions,inverse_class_weighting)
        fig.suptitle(title,fontsize=18)
        fig.tight_layout(rect=(0,0,1,.95))
    return fig


def render_weighted_unweighted_sets(parent_out, disease):
    """Matching three-figure external sets, drawn from their own saved predictions."""
    from matplotlib.backends.backend_pdf import PdfPages
    parent_out=Path(parent_out)
    outputs={}
    for weighted in (False,True):
        label='inverse_class_weighted' if weighted else 'unweighted'
        source=parent_out/'inverse_class_weighted' if weighted else parent_out
        if not (source/'data/classifier_patient_predictions.tsv').exists():
            continue
        out=parent_out/label
        (out/'plots').mkdir(parents=True,exist_ok=True)
        predictions=pd.read_csv(source/'data/classifier_patient_predictions.tsv',sep='\t')
        metrics=pd.read_csv(source/'data/classifier_performance.tsv',sep='\t')
        bundle=out/'external_classifier_plot_set.pdf'
        with plt.rc_context(STYLE),PdfPages(bundle) as pdf:
            for kind in ('ROC','PRC'):
                fig=single_curve_figure(predictions,metrics,disease,'external_transfer_selected_cohort',kind,weighted)
                pdf.savefig(fig,bbox_inches='tight',facecolor='white')
                save_publication(fig,out,'external_transfer_selected_cohort_classifier_'+kind)
            fig=confusion_figure(predictions,disease,'external_transfer_selected_cohort',weighted,metrics.groupby('panel').n_features.first().to_dict())
            pdf.savefig(fig,bbox_inches='tight',facecolor='white')
            save_publication(fig,out,'external_transfer_selected_cohort_classifier_confusion_matrices')
        title='Inverse class weighting' if weighted else 'No class weighting'
        (out/'PLOT_SET.md').write_text(
            f'# {disease}: {title}\n\n'
            '[Complete set: ROC, PRC, confusion matrices](external_classifier_plot_set.pdf)\n\n'
            'All five classifiers use seed-only and candidate-only panels. TabPFN3 and TabICLv2 use training-prior probability correction in the weighted set because their fit APIs do not support sample weights. The same external patients, fixed 0.50 classification threshold and plot styles are used in both sets.\n\n'
            f'Source predictions: {source / "data/classifier_patient_predictions.tsv"}\n'
            f'Source metrics: {source / "data/classifier_performance.tsv"}\n')
        outputs[label]=str(bundle)
    (parent_out/'classifier_plot_sets.json').write_text(json.dumps(outputs,indent=2)+'\n')
    return outputs


def compare_patient_classifiers(original, ometa, external, emeta, panels, out, disease, inverse_class_weighting=False):
    """Same fixed panels/patients/folds; no external fitting or tuning."""
    parent_out=Path(out)
    out=parent_out/'inverse_class_weighted' if inverse_class_weighting else parent_out
    for name in ('data','plots','models'):(out/name).mkdir(parents=True,exist_ok=True)
    y=ometa.loc[original.index,'label'].to_numpy(int);ey=emeta.loc[external.index,'label'].to_numpy(int)
    splits=list(StratifiedKFold(5,shuffle=True,random_state=RANDOM_SEED).split(original,y))
    folds=np.zeros(len(y),int)
    for k,(_,test) in enumerate(splits):folds[test]=k+1
    rows=[];records=[];model_files={};weight_records=[]
    for panel in ('seeds_only','candidates_only'):
        cols=list(panels[panel])
        if not cols or len(cols)!=len(set(cols)):
            raise ValueError(f'{panel}: empty or duplicated features')
        if set(cols)-set(original.columns) or set(cols)-set(external.columns):
            raise ValueError(f'{panel}: unavailable methylation features')
        for name in MODELS:
            print(f'{disease}: {panel} / {name}: 5 original-patient folds and external transfer',flush=True)
            oof=np.full(len(y),np.nan)
            for fold_number,(train,test) in enumerate(splits,1):
                fitted=fit_patient_model(name,original.iloc[train][cols],y[train],inverse_class_weighting)
                if inverse_class_weighting:
                    _,counts,weights=balanced_training_weights(y[train])
                    weight_records.extend(dict(panel=panel,model=name,scope='original_CV_training',fold=fold_number,
                                               label=k,n_patients=int(counts[k]),class_weight=float(weights[k]),
                                               total_class_weight=float(counts[k]*weights[k])) for k in (0,1))
                oof[test]=fitted.predict_proba(original.iloc[test][cols])[:,1]
            fitted=fit_patient_model(name,original[cols],y,inverse_class_weighting)
            if inverse_class_weighting:
                _,counts,weights=balanced_training_weights(y)
                weight_records.extend(dict(panel=panel,model=name,scope='final_original_training',fold=0,
                                           label=k,n_patients=int(counts[k]),class_weight=float(weights[k]),
                                           total_class_weight=float(counts[k]*weights[k])) for k in (0,1))
            ep=fitted.predict_proba(external[cols])[:,1]
            filename=f'{panel}_{name.lower().replace(" ","_")}.joblib'
            joblib.dump(fitted,out/'models'/filename,compress=3)
            model_files[f'{panel}/{name}']=dict(path='models/'+filename,features=cols)
            for cohort,labels,probabilities,meta,fold in [
                ('original_patient_CV',y,oof,ometa,folds),
                ('external_transfer_selected_cohort',ey,ep,emeta,np.zeros(len(ey),int))]:
                records.append(dict(panel=panel,model=name,cohort=cohort,n_features=len(cols),n_patients=len(labels),
                                    threshold=.5,**score(labels,probabilities),**classification_metrics(labels,probabilities),
                                    **bootstrap_metrics(labels,probabilities)))
                rows.extend(dict(sample_id=sample,panel=panel,model=name,cohort=cohort,label=int(label),
                                 probability=float(p),predicted_label_at_0_5=int(p>=.5),fold=int(f),series=study)
                            for sample,label,p,f,study in zip(meta.index,labels,probabilities,fold,meta.series))
    predictions=pd.DataFrame(rows);metrics=pd.DataFrame(records)
    # New logistic-regression predictions must reproduce the previous analysis.
    previous=out/'data/patient_predictions.tsv'
    if previous.exists() and not inverse_class_weighting:
        old=pd.read_csv(previous,sep='\t')
        old=old.loc[old.panel.isin(['seeds_only','candidates_only'])]
        keys=['sample_id','panel','cohort']
        check=predictions.loc[predictions.model=='Logistic regression']
        merged=old.merge(check,on=keys,suffixes=('_old','_new'),validate='one_to_one')
        if len(merged)!=len(old) or not np.allclose(merged.probability_old,merged.probability_new,rtol=0,atol=1e-12):
            raise ValueError('Logistic-regression predictions no longer match the original patient analysis')
    predictions.to_csv(out/'data/classifier_patient_predictions.tsv',sep='\t',index=False)
    metrics.to_csv(out/'data/classifier_performance.tsv',sep='\t',index=False)
    if weight_records:pd.DataFrame(weight_records).to_csv(out/'data/training_class_weights.tsv',sep='\t',index=False)
    pairs=[]
    for cohort in predictions.cohort.unique():
        for panel in ('seeds_only','candidates_only'):
            data=predictions.loc[(predictions.cohort==cohort)&(predictions.panel==panel)]
            pivot=data.pivot(index='sample_id',columns='model',values='probability')
            labels=data.drop_duplicates('sample_id').set_index('sample_id').loc[pivot.index,'label'].to_numpy(int)
            for left,right in combinations(MODELS,2):
                a,b=pivot[left].to_numpy(),pivot[right].to_numpy()
                rng=np.random.default_rng(RANDOM_SEED);deltas=[]
                for _ in range(500):
                    idx=np.r_[rng.choice(np.where(labels==0)[0],sum(labels==0),replace=True),
                              rng.choice(np.where(labels==1)[0],sum(labels==1),replace=True)]
                    deltas.append(roc_auc_score(labels[idx],b[idx])-roc_auc_score(labels[idx],a[idx]))
                pairs.append(dict(panel=panel,cohort=cohort,reference=left,comparison=right,
                                  delta_AUROC=roc_auc_score(labels,b)-roc_auc_score(labels,a),
                                  delta_AUROC_CI95_low=np.quantile(deltas,.025),delta_AUROC_CI95_high=np.quantile(deltas,.975),
                                  probability_correlation=np.corrcoef(a,b)[0,1],
                                  class_agreement_at_0_5=float(((a>=.5)==(b>=.5)).mean())))
    pd.DataFrame(pairs).to_csv(out/'data/classifier_pairwise_comparisons.tsv',sep='\t',index=False)
    for cohort in predictions.cohort.unique():
        fig=comparison_figure(predictions,metrics,disease,cohort,inverse_class_weighting)
        save_publication(fig,out,cohort+'_classifier_comparison')
        if cohort=='external_transfer_selected_cohort':
            fig=comparison_figure(predictions,metrics,disease,cohort,inverse_class_weighting)
            save_publication(fig,out,'external_transfer_selected_cohort_performance')
    save_publication(confusion_figure(predictions,disease,inverse_class_weighting=inverse_class_weighting,feature_counts=metrics.groupby('panel').n_features.first().to_dict()),out,'external_classifier_confusion_matrices')
    for cohort in predictions.cohort.unique():
        for kind in ('ROC','PRC'):
            save_publication(single_curve_figure(predictions,metrics,disease,cohort,kind,inverse_class_weighting),
                             out,cohort+'_classifier_'+kind)
        save_publication(confusion_figure(predictions,disease,cohort,inverse_class_weighting,metrics.groupby('panel').n_features.first().to_dict()),out,
                         cohort+'_classifier_confusion_matrices')
    manifest=dict(models=PARAMETERS,features={name:list(panels[name]) for name in ('seeds_only','candidates_only')},
                  training='Original-cohort patients only; fixed hyperparameters, no external fitting/tuning/early stopping',
                  preprocessing='Training-only median imputation for all models; training-only standardization for logistic regression',
                  original_CV='Same five stratified patient folds for all models, random seed '+str(RANDOM_SEED),
                  class_weights=('Per-fit original-training weights w_k=N_train/(2*n_k); identical sample_weight treatment for all three models; no double weighting or external-derived weights'
                                 if inverse_class_weighting else 'No class weighting or oversampling; XGBoost scale_pos_weight=1'),
                  inverse_class_weighting=inverse_class_weighting,
                  threshold=.5,intervals='Patient bootstrap conditional on fixed predictions and selected features; not selection-adjusted',
                  saved_models=model_files,versions={name:version(name) for name in ['scikit-learn','xgboost','joblib']},caution=CAUTION)
    if inverse_class_weighting:
        _,counts,weights=balanced_training_weights(y)
        manifest['final_original_class_weights']={str(k):dict(n=int(counts[k]),weight=float(weights[k])) for k in (0,1)}
    (out/'classifier_comparison_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    lines=[f'# {disease}: three-classifier patient comparison','',CAUTION,'',
           'Models use identical seed-only and candidate-only features. All final models are fitted on original patients and transferred unchanged to external patients.',
           'The original cohort is also evaluated with identical five-fold stratified patient CV. Hyperparameters are fixed before external evaluation; no external tuning or early stopping is performed.',
           '', '| Cohort | Panel | Model | CpGs | AUROC | AUPRC | Brier | Balanced accuracy at 0.50 |',
           '|---|---|---|---:|---:|---:|---:|---:|']
    for row in metrics.itertuples():
        lines.append(f'| {row.cohort} | {row.panel} | {row.model} | {row.n_features} | {row.AUROC:.3f} | {row.AUPRC:.3f} | {row.Brier:.3f} | {row.balanced_accuracy_at_0_5:.3f} |')
    lines += ['', 'Probability predictions, fixed-threshold patient classes, confidence intervals, confusion counts and paired model comparisons are in data/classifier_*.tsv.',
              'Bootstrap AUROC-difference intervals use the same sampled patients for each pair of models. They do not account for panel selection, model refitting or study confounding.',
              'The 0.50 classification threshold is fixed, not optimized on external patients. Confusion matrices report counts and row percentages.',
              'Fitted preprocessing/classifier pipelines are saved in models/. Use the feature order in classifier_comparison_manifest.json. Load only trusted joblib artifacts.',
              '', '## Figures','',
              '- [External ROC and PR model comparison](plots/external_transfer_selected_cohort_performance.pdf)',
              '- [Original-patient CV model comparison](plots/original_patient_CV_classifier_comparison.pdf)',
              '- [External confusion matrices](plots/external_classifier_confusion_matrices.pdf)']
    if inverse_class_weighting:
        lines += ['', '## Inverse class weighting', '',
                  'Every fit uses training-only weights w_k=N_train/(2*n_k). Weights are recomputed in each CV fold; final weights use all original patients. Only classifier loss/splits are weighted; imputation and logistic standardization remain training-only, unweighted transforms.',
                  'The same sample-weight formula is passed to all three estimators. Built-in class weights remain disabled and XGBoost scale_pos_weight stays 1, avoiding double weighting.',
                  'External labels never determine training weights. External metrics and confusion counts are unweighted patient evaluations. Threshold 0.50 is fixed; weighted scores are not automatically population-calibrated disease probabilities.',
                  'Final original-cohort weights: '+json.dumps(manifest['final_original_class_weights']),
                  '', 'Separate external figures:',
                  '- [ROC](plots/external_transfer_selected_cohort_classifier_ROC.pdf)',
                  '- [PRC](plots/external_transfer_selected_cohort_classifier_PRC.pdf)',
                  '- [Confusion matrices](plots/external_transfer_selected_cohort_classifier_confusion_matrices.pdf)']
    (out/'CLASSIFIER_COMPARISON.md').write_text('\n'.join(lines)+'\n')
    report=parent_out/'REPORT.md'
    if report.exists():
        text=report.read_text()
        if '[Three-classifier comparison]' not in text:
            text+='\n## Patient classifier comparison\n\n[Three-classifier comparison](CLASSIFIER_COMPARISON.md): logistic regression, Random Forest and XGBoost using the same seed-only and candidate-only panels.\n'
        if inverse_class_weighting and '[Inverse-class-weighted model comparison]' not in text:
            text+='\n[Inverse-class-weighted model comparison](inverse_class_weighted/CLASSIFIER_COMPARISON.md): original-training inverse-frequency sample weights, new ROC/PRC/confusion figures.\n'
        report.write_text(text)
    render_weighted_unweighted_sets(parent_out,disease)
    return metrics,predictions


def main(root):
    from omegaconf import OmegaConf
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--inverse-class-weighting',action='store_true',help='Training-only N/(2*n_class) weights; save a separate inverse_class_weighted analysis')
    args=parser.parse_args();config=OmegaConf.load(args.config)
    base=Path(config.p_base).resolve();out=base/'accumulated_results/post_evaluation'
    stamp=json.loads((out/'cache/fingerprint.json').read_text())
    for path,size,mtime in stamp['inputs']['files']:
        p=Path(path)
        if (p.stat().st_size,p.stat().st_mtime_ns)!=(size,mtime):
            raise ValueError('Cached raw data are stale; run --stage post-evaluation first')
    original=pd.read_pickle(out/'cache/original.pkl');external=pd.read_pickle(out/'cache/external.pkl')
    ometa=original_metadata(config,original)
    emeta=pd.read_csv(out/'cache/external_metadata.tsv',sep='\t',index_col=0).loc[external.index]
    panels={name:pd.read_csv(out/'data'/f'{name}_features.tsv',sep='\t').CpG_ID.tolist()
            for name in ('seeds_only','candidates_only')}
    disease='BC' if Path(root).name=='breast_cancer_analysis' else 'AML'
    compare_patient_classifiers(original,ometa,external,emeta,panels,out,disease,args.inverse_class_weighting)
    if (out/'data/classifier_patient_predictions.tsv').exists() and not args.inverse_class_weighting:
        from .foundations import augment_foundation_models
        augment_foundation_models(root,args.config)
    destination=out/'inverse_class_weighted' if args.inverse_class_weighting else out
    print(f'{disease}: patient comparison complete -> {destination}',flush=True)
