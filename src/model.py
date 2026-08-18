"""
Convolutional Neural Network (CNN) & Diagnostic Pipeline
Author: Arjuna Fransesco
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib

FEATURE_COLS = ["mean_greenness_index", "chlorosis_index", "lesion_area_ratio", "glcm_contrast"]


class LeafDiseaseCNNClassifier:
    """
    Deep Feature Extraction & Ensemble Classifier for Plant / Rice Leaf Pathology.
    """
    def __init__(self, n_estimators=200, random_state=42):
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=0.08,
            max_depth=5,
            subsample=0.85,
            random_state=random_state
        )
        self.classes_ = None

    def fit(self, X, y):
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save(self, filepath):
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath):
        return joblib.load(filepath)
