import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from analyze import CATEGORICAL, FEATURES, NUMERIC, make_pipeline, probability_metrics


class MethodologyTests(unittest.TestCase):
    def test_heldout_extreme_does_not_change_training_scaler_or_encoder(self):
        frame = pd.DataFrame({name: [0., 1., 2., 3., 4., 5.] for name in NUMERIC})
        for name in CATEGORICAL:
            frame[name] = ["a", "a", "b", "b", "a", "b"]
        pipeline = make_pipeline(LogisticRegression())
        pipeline.fit(frame, [0, 1, 0, 1, 0, 1])
        numeric = pipeline.named_steps["preprocess"].named_transformers_["numeric"]
        before = numeric.named_steps["scale"].data_max_.copy()
        heldout = frame.iloc[[0]].copy()
        heldout[NUMERIC] = 10000.
        heldout[CATEGORICAL] = "unseen-category"
        pipeline.predict_proba(heldout)
        np.testing.assert_array_equal(numeric.named_steps["scale"].data_max_, before)
        self.assertTrue(np.all(before == 5))

    def test_ap_uses_probability_ranking_and_is_not_mislabeled_roc_auc(self):
        actual = np.array([0, 1, 0, 1])
        probabilities = np.array([.1, .9, .6, .7])
        result = probability_metrics(actual, probabilities)
        self.assertEqual(result["average_precision"], 1.0)
        self.assertNotEqual(result["average_precision"], average_precision_score(actual, probabilities >= .5))
        self.assertIn("roc_auc", result)
        self.assertAlmostEqual(result["positive_recall"], 1.0)

    def test_identifier_and_notebook_artifacts_are_outside_feature_contract(self):
        self.assertEqual(len(FEATURES), 10)
        self.assertTrue(set(FEATURES).isdisjoint({"#", "RowNumber", "CustomerId", "Surname", "NumOfProducts_int", "Exited"}))


if __name__ == "__main__":
    unittest.main()
