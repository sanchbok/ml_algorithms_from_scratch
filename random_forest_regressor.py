"""
Custom random forest regressor
"""
import random
from typing import Literal

import pandas as pd

from decision_tree_regressor import MyTreeReg


class MyForestReg:
    """
    Custom random forest regressor
    """
    def __init__(
        self,
        n_estimators: int = 10,
        max_features: float = 0.5,
        max_samples: float = 0.5,
        random_state: int = 42,
        max_depth: int = 5,
        min_samples_split: int = 2,
        max_leafs: int = 20,
        bins: int = 16,
        oob_score: Literal['mae', 'mse', 'rmse', 'mape', 'r2'] = None
    ) -> None:
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_samples = max_samples
        self.random_state = random_state
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins
        self.oob_score = oob_score
        self.oob_score_ = 0
        self.leafs_cnt = 0
        self.trees = []

    def __str__(self) -> str:
        return (
            f'MyForestReg class: n_estimators={self.n_estimators}, '
            f'max_features={self.max_features}, '
            f'max_samples={self.max_samples}, '
            f'max_depth={self.max_depth}, '
            f'min_samples_split={self.min_samples_split}, '
            f'max_leafs={self.max_leafs}, '
            f'bins={self.bins}, '
            f'random_state={self.random_state}'
        )

    def _get_oob_score(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        if self.oob_score == 'mae':
            return (y_true - y_pred).abs().mean()
        if self.oob_score == 'mse':
            return ((y_true - y_pred) ** 2).mean()
        if self.oob_score == 'rmse':
            return ((y_true - y_pred) ** 2).mean()**0.5
        if self.oob_score == 'mape':
            return 100 * ((y_true - y_pred)/y_true).abs().mean()
        if self.oob_score == 'r2':
            return 1 - ((y_true - y_pred)**2).sum()/((y_true - y_true.mean())**2).sum()
        raise ValueError('Unknown oob_score')

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Fit random forest
        """
        random.seed(self.random_state)
        self.fi = {col: 0 for col in X.columns}

        oob_predictions = pd.Series([])

        for _ in range(self.n_estimators):
            cols_idx = random.sample(X.columns.tolist(), round(self.max_features * X.shape[1]))
            rows_idx = random.sample(range(X.shape[0]), round(self.max_samples * X.shape[0]))

            X_sample = X.reset_index(drop=True)[cols_idx].iloc[rows_idx]
            y_sample = y.reset_index(drop=True)[rows_idx]

            tree = MyTreeReg(
                self.max_depth, self.min_samples_split, self.max_leafs, self.bins, X.shape[0]
            )
            tree.fit(X_sample, y_sample)

            if self.oob_score:
                oob_idx = list(set(range(X.shape[0])) - set(rows_idx))
                X_sample_oob = X.reset_index(drop=True)[cols_idx].iloc[oob_idx]
                oob_predictions = pd.concat([oob_predictions, tree.predict(X_sample_oob)])

            self.trees.append(tree)
            self.leafs_cnt += tree.leafs_cnt

            for col in X_sample.columns:
                self.fi[col] += tree.fi[col]

        if self.oob_score:
            oob_predictions = oob_predictions.groupby(level=0).mean().sort_index()
            self.oob_score_ = self._get_oob_score(
                y[oob_predictions.index], oob_predictions
            )

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict target values
        """
        predictions = pd.Series([0 for _ in range(X.shape[0])], index=X.index)
        for tree in self.trees:
            prediction = tree.predict(X)
            predictions += prediction
        return predictions/self.n_estimators
