"""
Custom bagging classifier algorithm
"""
import random
from copy import copy
from typing import Literal, Type

import numpy as np
import pandas as pd


class MyBaggingClf:
    """
    Custom bagging classifier algorithm
    """
    def __init__(
        self,
        estimator: Type = None,
        n_estimators: int = 10,
        max_samples: float = 1.0,
        random_state: int = 42,
        oob_score: Literal['accuracy', 'precision', 'recall', 'f1', 'roc_auc'] = None
    ) -> None:
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.estimators = []
        self.oob_score = oob_score
        self.oob_score_ = 0

    def __str__(self) -> str:
        return (
            f'MyBaggingClf class: estimator={self.estimator}, '
            f'n_estimators={self.n_estimators}, max_samples={self.max_samples}, '
            f'random_state={self.random_state}'
        )

    def _get_metric(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        if self.oob_score == 'roc_auc':
            total = pd.DataFrame({'true': y_true, 'pred': y_pred}).sort_values(
                by='pred', ascending=False
            )
            score = 0
            for _, row in total.iterrows():
                if row.true == 0:
                    higher = sum(total.loc[total.pred > row.pred, 'true'] > 0)
                    equal = sum(total.loc[total.pred == row.pred, 'true'] > 0)/2
                    score += higher + equal
            return score/(sum(y_true == 1) * sum(y_true == 0))
        
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0

        if self.oob_score == 'accuracy':
            return sum(y_true == y_pred)/len(y_true)

        if self.oob_score == 'precision':
            return sum(y_pred[y_true == y_pred] == 1)/sum(y_pred == 1)

        if self.oob_score == 'recall':
            return sum(y_pred[y_true == y_pred] == 1)/sum(y_true)

        if self.oob_score == 'f1':
            precision = sum(y_pred[y_true == y_pred] == 1)/sum(y_pred == 1)
            recall = sum(y_pred[y_true == y_pred] == 1)/sum(y_true)
            return 2 * precision * recall/(precision + recall)

        raise ValueError('Unknown oob_score')

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Fit algorithm
        """
        random.seed(self.random_state)
        sample_rows_idx = []
        oob_rows_idx = []
        oob_predictions = pd.Series([])

        for _ in range(self.n_estimators):
            rows_idx = random.choices(
                range(round(X.shape[0])), k=round(X.shape[0] * self.max_samples)
            )
            sample_rows_idx.append(rows_idx)

            if self.oob_score:
                oob_idx = np.setdiff1d(np.arange(X.shape[0]), rows_idx)
                oob_rows_idx.append(oob_idx)

        for id, idx in enumerate(sample_rows_idx):
            model = copy(self.estimator)
            model.fit(X.reset_index(drop=True).iloc[idx], y.reset_index(drop=True)[idx])

            if self.oob_score:
                current_predictions = model.predict_proba(
                    X.reset_index(drop=True).iloc[oob_rows_idx[id]]
                )
                if isinstance(current_predictions, np.ndarray):
                    current_predictions = pd.Series(current_predictions, index=oob_rows_idx[id])
                oob_predictions = pd.concat([oob_predictions, current_predictions])

            self.estimators.append(model)

        if self.oob_score:
            oob_predictions = oob_predictions.groupby(level=0).mean()
            self.oob_score_ = self._get_metric(
                y[oob_predictions.index], oob_predictions
            )

    def predict(self, X: pd.DataFrame, type: Literal['mean', 'vote']) -> pd.Series:
        """
        Predict target lables
        """
        if type == 'mean':
            probabilities = self.predict_proba(X)
            probabilities[probabilities > 0.5] = 1
            probabilities[probabilities <= 0.5] = 0
            return probabilities.astype(int)

        if type == 'vote':
            final_predictions = pd.Series([])
            for model in self.estimators:
                predictions = model.predict(X)
                if isinstance(predictions, np.ndarray):
                    predictions = pd.Series(predictions, index=X.index)
                final_predictions = pd.concat([final_predictions, predictions])
            return final_predictions.groupby(level=0).agg(pd.Series.mode).sort_index()

        raise ValueError('Unknown type')

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict positive label probabilities
        """
        probabilities = np.zeros(X.shape[0])
        for model in self.estimators:
            probabilities += np.asarray(model.predict_proba(X))
        return pd.Series(probabilities/self.n_estimators, index=X.index)
