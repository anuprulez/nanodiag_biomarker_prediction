"""Check the explicitly different weighting method for pretrained classifiers."""
import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.model_selection import StratifiedKFold
from .analysis import RANDOM_SEED
from .classifiers import balanced_training_weights
from .foundations import prior_correct,weighted_foundation_predictions

class FoundationTests(unittest.TestCase):
    def test_prior_correction_preserves_probability_sum_and_external_ranking(self):
        p=np.array([.1,.3,.6,.8]);weights=np.array([.625,2.5])
        corrected=prior_correct(np.column_stack([1-p,p]),weights)
        np.testing.assert_allclose(corrected.sum(axis=1),1.)
        np.testing.assert_allclose(corrected[:,1]/corrected[:,0],(p/(1-p))*4)
        y=np.array([0,1,0,1])
        self.assertEqual(roc_auc_score(y,p),roc_auc_score(y,corrected[:,1]))
        self.assertEqual(average_precision_score(y,p),average_precision_score(y,corrected[:,1]))

    def test_oof_correction_uses_training_fold_priors_not_external_labels(self):
        labels=np.r_[np.zeros(43,int),np.ones(11,int)]
        raw=pd.DataFrame({'fold':[0,1,2,3,4,5],'probability':[.2]*6,
                          'predicted_label_at_0_5':[0]*6,'label':[0,1,1,0,1,0]})
        result=weighted_foundation_predictions(raw,labels)
        splits=list(StratifiedKFold(5,shuffle=True,random_state=RANDOM_SEED).split(np.zeros(len(labels)),labels))
        for fold in range(6):
            training=labels if fold==0 else labels[splits[fold-1][0]]
            weights=balanced_training_weights(training)[2]
            expected=prior_correct([[.8,.2]],weights)[0,1]
            self.assertAlmostEqual(result.probability.iloc[fold],expected)
        raw['label']=1-raw.label
        np.testing.assert_array_equal(result.probability,weighted_foundation_predictions(raw,labels).probability)

if __name__=='__main__':unittest.main()
