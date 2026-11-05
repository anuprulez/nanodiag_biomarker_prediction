"""Patient preprocessing isolation and fixed-threshold classification checks."""
import unittest
import numpy as np
from .classifiers import MODELS, patient_estimator, classification_metrics, balanced_training_weights, fit_patient_model

class ClassifierTests(unittest.TestCase):
    def test_training_imputation_is_not_refitted_on_external_patients(self):
        x=np.array([[0.,np.nan],[.1,.2],[.8,.8],[1.,1.]])
        y=np.array([0,0,1,1])
        for name in MODELS:
            with self.subTest(model=name):
                model=patient_estimator(name)
                if name!='Logistic regression':model[-1].set_params(n_estimators=5,n_jobs=1)
                model.fit(x,y)
                before=model[0].statistics_.copy()
                np.testing.assert_allclose(before,[.45,.8])
                p=model.predict_proba(np.array([[np.nan,.01],[.99,np.nan]]))[:,1]
                self.assertTrue(np.isfinite(p).all() and ((p>=0)&(p<=1)).all())
                np.testing.assert_array_equal(model[0].statistics_,before)

    def test_inverse_weights_balance_training_classes_and_use_each_fold_counts(self):
        labels=np.array([0]*8+[1]*2)
        weights,counts,classes=balanced_training_weights(labels)
        np.testing.assert_allclose(classes,[.625,2.5])
        self.assertAlmostEqual(weights.mean(),1.)
        self.assertAlmostEqual(weights[labels==0].sum(),5.)
        self.assertAlmostEqual(weights[labels==1].sum(),5.)
        _,fold_counts,fold_weights=balanced_training_weights(np.array([0]*6+[1]))
        np.testing.assert_allclose(fold_counts,[6,1])
        np.testing.assert_allclose(fold_weights,[7/12,7/2])
        with self.assertRaisesRegex(ValueError,'both training classes'):
            balanced_training_weights(np.zeros(5,dtype=int))

    def test_sample_weights_change_logistic_loss_without_external_labels(self):
        x=np.zeros((100,1));labels=np.array([0]*80+[1]*20)
        ordinary=fit_patient_model('Logistic regression',x,labels)
        weighted=fit_patient_model('Logistic regression',x,labels,True)
        self.assertLess(ordinary.predict_proba([[0.]])[0,1],.3)
        self.assertAlmostEqual(weighted.predict_proba([[0.]])[0,1],.5,places=6)

    def test_confusion_counts_use_prespecified_half_threshold(self):
        result=classification_metrics(np.array([0,0,1,1]),np.array([.2,.5,.49,.9]))
        self.assertEqual([result[k] for k in ['TN','FP','FN','TP']],[1,1,1,1])
        self.assertEqual(result['balanced_accuracy_at_0_5'],.5)

if __name__=='__main__':unittest.main()
