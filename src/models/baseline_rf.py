from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score)
import joblib


class BaselineRFModel:
    def __init__(self, n_estimators=100, random_state=42):
        """
        Multilabel Random Forest Classifier for Biosignatures.
        """
        self.model = MultiOutputClassifier(
            RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=random_state,
                n_jobs=-1,
                class_weight='balanced'))

    def train(self, X_train, y_train):
        """
        Train the model. X_train: DataFrame of features.
        y_train: DataFrame of binary labels.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        """
        Returns an array of shape (n_samples, n_classes)
        containing probabilities.
        """
        raw_probas = self.model.predict_proba(X_test)
        # MultiOutputClassifier predict_proba returns a list of arrays
        # for each class
        # Convert to standard shape: (n_samples, n_classes)
        import numpy as np
        probas = np.array([p[:, 1] for p in raw_probas]).T
        return probas

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        probas = self.predict_proba(X_test)

        metrics = {
            'precision': precision_score(y_test, y_pred, average='macro'),
            'recall': recall_score(y_test, y_pred, average='macro'),
            'f1': f1_score(y_test, y_pred, average='macro'),
            'roc_auc': roc_auc_score(y_test, probas, average='macro')
        }
        return metrics

    def save(self, filepath):
        joblib.dump(self.model, filepath)

    def load(self, filepath):
        self.model = joblib.load(filepath)
