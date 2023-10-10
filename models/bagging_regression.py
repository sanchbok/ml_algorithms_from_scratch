"""
Custom bagging regression algorithm
"""
import random
from copy import copy
from typing import Literal, Type

import numpy as np
import pandas as pd


class MyBaggingReg:
    """
    Custom bagging regressor
    """
    def __init__(
        self,
        estimator: Type = None,
        n_estimators: int = 10,
        max_samples: float = 1.0,
        random_state: int = 42,
        oob_score: Literal['mae', 'mse', 'rmse', 'mape', 'r2'] = None
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
            f"MyBaggingReg class: estimator={self.estimator}, "
            f"n_estimators={self.n_estimators}, max_samples={self.max_samples}, "
            f"random_state={self.random_state}"
        )

    def _get_metric(self, y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
        """ 
        Compute metric
        """
        if self.oob_score == 'mae':
            return (y_true - y_pred).abs().mean()

        if self.oob_score == 'mse':
            return ((y_true - y_pred)**2).mean()

        if self.oob_score == 'rmse':
            return ((y_true - y_pred)**2).mean()**0.5

        if self.oob_score == 'mape':
            return ((y_true - y_pred)/y_true).abs().mean() * 100

        if self.oob_score == 'r2':
            return 1 - ((y_true - y_pred)**2).sum()/((y_true - y_true.mean())**2).sum()

        raise ValueError('Unknown metric')

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Fit ensemble
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

        for i, idx in enumerate(sample_rows_idx):
            model = copy(self.estimator)
            model.fit(X.reset_index(drop=True).iloc[idx], y.reset_index(drop=True)[idx])

            if self.oob_score:
                predictions = model.predict(
                    X.reset_index(drop=True).iloc[oob_rows_idx[i]]
                )

                if isinstance(predictions, np.ndarray):
                    predictions = pd.Series(predictions, index=oob_rows_idx[i])
                oob_predictions = pd.concat([oob_predictions, predictions])

            self.estimators.append(model)

        if self.oob_score:
            oob_predictions = oob_predictions.groupby(level=0).mean()
            self.oob_score_ = self._get_metric(
                y.reset_index(drop=True)[oob_predictions.index], oob_predictions
            )

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict target values
        """
        predictions = np.zeros(len(X))
        for model in self.estimators:
            predictions += np.asarray(model.predict(X))
        return predictions/self.n_estimators
