#!/usr/bin/env python3
"""Plot GSE52865 transfer as A and within-cohort cross-validation as B."""
from pathlib import Path
import hashlib
import json
import shutil
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from pypdf import PdfReader, PdfWriter
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from post_evaluation_support.publication import STYLE, save_publication
from post_evaluation_support.candidate_comparison import MODELS


def main():
    parent = ROOT/'data_corr_0.33_knn_n_3000_n_neg_10000/accumulated_results/post_evaluation'
    out = ROOT/'evaluation/independent_classification/GSE52865'
    original_path = parent/'data/classifier_patient_predictions.tsv'
    independent_path = parent/'independent_feature_verification/data/patient_predictions.tsv'
    previous = pd.read_csv(original_path, sep='\t')
    independent = pd.read_csv(independent_path, sep='\t')
    assert set(independent.series) == {'GSE52865'}
    independent['cohort'] = 'independent_GSE52865'
    cv_predictions = pd.read_csv(out/'cross_validation/out_of_fold_predictions.tsv', sep='\t')
    predictions = pd.concat([previous, independent, cv_predictions], ignore_index=True)
    predictions = predictions.loc[predictions.panel.eq('candidates_only') & predictions.model.isin([m for m,_ in MODELS])]
    cohorts = [('independent_GSE52865', 'A  Train on original cohort, predict on independent GSE52865', '#009E73'),
               ('GSE52865_patient_CV', 'B  Independent GSE52865: five-fold cross-validation', '#7B2CBF')]
    records = []
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(17,10.5))
        grid = fig.add_gridspec(2,len(MODELS))
        axes = np.array([[fig.add_subplot(grid[row,col]) for col in range(len(MODELS))] for row in range(2)])
        fig.subplots_adjust(left=.085,right=.90,bottom=.12,top=.83,wspace=.90,hspace=.95)
        for row,(cohort,heading,color) in enumerate(cohorts):
            cmap = LinearSegmentedColormap.from_list(cohort,['#ffffff',color])
            for col,(model,label) in enumerate(MODELS):
                frame = predictions.loc[predictions.model.eq(model) & predictions.cohort.eq(cohort)]
                assert len(frame) and not frame.sample_id.duplicated().any()
                assert np.array_equal(frame.predicted_label_at_0_5, (frame.probability >= .5).astype(int))
                if cohort in {'independent_GSE52865', 'GSE52865_patient_CV'}:
                    assert len(frame) == 57 and frame.label.sum() == 40
                matrix = confusion_matrix(frame.label,frame.predicted_label_at_0_5,labels=[0,1])
                fractions = matrix/matrix.sum(axis=1,keepdims=True)
                ax = axes[row,col]
                image = ax.imshow(fractions,cmap=cmap,vmin=0,vmax=1)
                for i in range(2):
                    for j in range(2):
                        ax.text(j,i,f'{matrix[i,j]:,}\n({fractions[i,j]:.1%})',ha='center',va='center',fontsize=15.5,
                                color='white' if fractions[i,j]>.6 else '#202020')
                        records.append(dict(panel='AB'[row],cohort=cohort,model=model,observed=i,predicted=j,count=int(matrix[i,j]),within_class_fraction=float(fractions[i,j])))
                labels = ['Normal','Breast cancer']
                ax.set_xticks([0,1],labels);ax.set_yticks([0,1],labels)
                ax.set_xlabel('Predicted class',fontsize=17,labelpad=8)
                ax.set_ylabel('Ground truth',fontsize=17,labelpad=8)
                ax.set_title(label,fontsize=20,pad=13)
                ax.tick_params(length=0,labelsize=14,pad=6)
                ax.tick_params(axis='x',labelrotation=25)
                for spine in ax.spines.values():spine.set_visible(False)
            fig.canvas.draw()
            box = axes[row,-1].get_position()
            fig.text(.50,box.y1+.075,heading,ha='center',fontsize=20)
            color_axis = fig.add_axes([.935,box.y0,.012,box.height])
            colorbar = fig.colorbar(image,cax=color_axis,ticks=[0,.5,1])
            colorbar.ax.set_yticklabels(['0%','50%','100%'])
            colorbar.set_label('Within-class percentage',fontsize=15,labelpad=10)
            colorbar.ax.tick_params(labelsize=13,length=3)
            colorbar.outline.set_visible(False)
        fig.suptitle('Breast cancer: 33 candidate CpGs',fontsize=25,y=.98)
        save_publication(fig,out,'GSE52865_parts_C_D_confusion_matrices')
    pd.DataFrame(records).to_csv(out/'GSE52865_parts_C_D_confusion_counts.tsv',sep='\t',index=False)
    manifest = dict(title='Breast cancer: 33 candidate CpGs',panels={letter:heading for letter,(_,heading,_) in zip('AB',cohorts)},
                    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [original_path,independent_path,out/'cross_validation/out_of_fold_predictions.tsv']},
                    n_features=33,threshold=.5,procedure='A reuses original-trained predictions; B uses stratified five-fold out-of-fold GSE52865 predictions with fixed features/settings, training-only imputation and no threshold tuning')
    (out/'GSE52865_parts_C_D_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(out/'plots/GSE52865_parts_C_D_confusion_matrices.pdf')
    print('Verified 8 confusion matrices; panels A and B each contain 57 specimens per model.')

if __name__ == '__main__':main()
