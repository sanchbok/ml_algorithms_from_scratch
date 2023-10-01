"""
Custom random forest classifier
"""
import random
from typing import Literal

import numpy as np
import pandas as pd

from decision_tree_classifier import MyTreeClf


class MyForestClf:
    """
    Custom random forest classifier
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
        criterion: Literal['entropy', 'gini'] = 'entropy',
        oob_score: Literal['accuracy', 'precision', 'recall', 'f1', 'roc_auc'] = None
    ) -> None:

        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_samples = max_samples
        self.random_state = random_state
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins
        self.criterion = criterion
        self.leafs_cnt = 0
        self.trees = []
        self.fi = {}
        self.oob_score = oob_score
        self.oob_score_ = 0

    def __str__(self) -> str:
        return (
            f"MyForestClf class: n_estimators={self.n_estimators}, "
            f"max_features={self.max_features}, max_samples={self.max_samples}, "
            f"max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, "
            f"max_leafs={self.max_leafs}, bins={self.bins}, criterion={self.criterion}, "
            f"random_state={self.random_state}"
        )

    def _get_oob_score(self, y_true: pd.Series, y_pred: pd.Series) -> float:
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
        Fit model
        """
        random.seed(self.random_state)
        self.fi = {col: 0 for col in X.columns}
        oob_predictions = pd.Series([])

        for _ in range(self.n_estimators):
            cols_idx = random.sample(X.columns.tolist(), round(self.max_features * X.shape[1]))
            rows_idx = random.sample(range(X.shape[0]), round(self.max_samples * X.shape[0]))

            X_sample = X.reset_index(drop=True)[cols_idx].iloc[rows_idx]
            y_sample = y.reset_index(drop=True)[rows_idx]

            tree = MyTreeClf(
                self.max_depth,
                self.min_samples_split,
                self.max_leafs,
                self.bins,
                self.criterion,
                X.shape[0]
            )

            tree.fit(X_sample, y_sample)

            if self.oob_score:
                oob_idx = np.setdiff1d(
                    np.arange(X.shape[0]), rows_idx
                )
                X_sample_oob = X.reset_index(drop=True)[cols_idx].iloc[oob_idx]
                oob_predictions = pd.concat([oob_predictions, tree.predict_proba(X_sample_oob)])

            self.trees.append(tree)
            self.leafs_cnt += tree.leafs_cnt

            for col in X_sample.columns:
                self.fi[col] += tree.fi[col]

        if self.oob_score:
            oob_predictions = oob_predictions.groupby(level=0).mean().sort_index()
            self.oob_score_ = self._get_oob_score(
                y[oob_predictions.index], oob_predictions
            )

    def predict(self, X: pd.DataFrame, type: Literal['mean', 'vote']) -> pd.Series:
        """
        Predict target labels
        """
        if type == 'mean':
            probabilties = self.predict_proba(X)
            probabilties[probabilties > 0.5] = 1
            probabilties[probabilties <= 0.5] = 0
            return probabilties.astype(int)

        if type == 'vote':
            forest_predictions = pd.Series([])
            for tree in self.trees:
                forest_predictions = pd.concat([forest_predictions, tree.predict(X)])
            return forest_predictions.groupby(level=0).agg(pd.Series.mode).sort_index()

        raise ValueError('Unkown type value')

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict target probabilities
        """
        forest_predictions = pd.Series(np.zeros(X.shape[0]), index=X.index)
        for tree in self.trees:
            forest_predictions += tree.predict_proba(X)
        return forest_predictions/self.n_estimators
