''' Custom decision tree classifier '''
from typing import Dict, List, Literal, Tuple

import numpy as np
import pandas as pd


class MyTreeClf:
    ''' Custom decision tree classifier class '''
    def __init__(
        self,
        max_depth: int = 5,
        min_samples_split: int = 2,
        max_leafs: int = 20,
        bins: int = None,
        criterion: Literal['entropy', 'gini'] = 'entropy',
        total_samples: int = None
    ) -> None:

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.leafs_cnt = 0
        self.bins = bins
        self.criterion = criterion
        self.total_samples = total_samples

    def __str__(self) -> str:
        return (
            f'MyTreeClf class: max_depth={self.max_depth}, '
            f'min_samples_split={self.min_samples_split}, '
            f'max_leafs={self.max_leafs}'
        )

    def _get_criterion(self, y: pd.Series) -> float:
        ''' Compute critetion '''
        if self.criterion == 'entropy':
            entropy = 0
            for label in y.unique():
                proba = sum(y == label)/len(y)
                if proba == 0:
                    continue
                entropy += proba * np.log2(proba)
            return -entropy

        if self.criterion == 'gini':
            impurity = 1
            for label in y.unique():
                proba = sum(y == label)/len(y)
                impurity -= proba**2
            return impurity

        raise ValueError('Unknown criterion')

    def _get_criterion_gain(self, y_left: pd.Series, y_right: pd.Series, criterion: float) -> float:
        ''' Compute criterion gain '''
        criterion_left, criterion_right = self._get_criterion(y_left), self._get_criterion(y_right)
        n_left, n_right = len(y_left), len(y_right)
        new_criterion = n_left/(n_left + n_right) * criterion_left + \
            n_right/(n_left + n_right) * criterion_right
        return criterion - new_criterion

    def _get_feature_importance(
        self,
        left_samples: pd.Series,
        right_samples: pd.Series,
    ) -> float:
        ''' Compute feature importance for tree node '''
        node_criterion = self._get_criterion(pd.concat([left_samples, right_samples]))
        criterion_gain = self._get_criterion_gain(left_samples, right_samples, node_criterion)
        node_num = len(left_samples) + len(right_samples)
        return node_num/self.total_samples * criterion_gain

    def _get_thresholds(self, X: pd.DataFrame) -> Dict[str, List[float]]:
        ''' Compute features' threshold values to split target '''
        thresholds = {}
        for col in X.columns:
            feature_values = sorted(X[col].unique())
            col_thresholds = []

            for i in range(len(feature_values) - 1):
                col_thresholds.append(
                    (feature_values[i] + feature_values[i + 1])/2
                )

            if self.bins is None or len(col_thresholds) < self.bins:
                thresholds[col] = col_thresholds
            else:
                thresholds[col] = np.histogram(feature_values, bins=self.bins)[1][1:-1]

        return thresholds

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        ''' Fit model '''

        def helper(
            X: pd.DataFrame,
            y: pd.Series,
            depth: int = 0,
            node_type: str = 'parent'
        ) -> dict:
            # Stop because of tree depth
            if depth == self.max_depth:
                return y.mean()

            # Stop because of min number of samples to split node
            if len(X) < self.min_samples_split:
                return y.mean()

            # Stop because y consists of one class
            if y.nunique() < 2:
                return y.mean()

            # Stop because of max number of leaves
            if self.leafs_cnt >= self.max_leafs:
                return y.mean()

            if node_type == 'child':
                self.leafs_cnt -= 1

            split_column, split_value, cg = self.get_best_split(X, y)
            X_left, y_left = X[X[split_column] <= split_value], y[X[split_column] <= split_value]
            X_right, y_right = X[X[split_column] > split_value], y[X[split_column] > split_value]

            fitted_tree = {(split_column, split_value, cg): {}}
            self.fi[split_column] += self._get_feature_importance(y_left, y_right)

            self.leafs_cnt += 2
            fitted_tree[(split_column, split_value, cg)]['left'] = \
                helper(X_left, y_left, depth + 1, 'child')
            fitted_tree[(split_column, split_value, cg)]['right'] = \
                helper(X_right, y_right, depth + 1, 'child')

            return fitted_tree

        self.fi = {col: 0 for col in X.columns}
        self.thresholds = self._get_thresholds(X)
        if not self.total_samples:
            self.total_samples = X.shape[0]
        self.fitted_tree = helper(X, y)

    def get_best_split(self, X: pd.DataFrame, y: pd.Series) -> Tuple[str, float, float]:
        ''' Find best column and threshold to split '''
        best_split_column, best_split_value, best_cg = 0, 0, 0
        criterion = self._get_criterion(y)

        for col in X.columns:
            for threshold in self.thresholds[col]:
                y_left, y_right = y[X[col] <= threshold], y[X[col] > threshold]
                current_cg = self._get_criterion_gain(y_left, y_right, criterion)
                if current_cg > best_cg:
                    best_split_column, best_split_value, best_cg = col, threshold, current_cg

        return best_split_column, best_split_value, best_cg

    def predict(self, X: pd.DataFrame) -> pd.Series:
        ''' Compute target labels '''
        probabilities = self.predict_proba(X)
        probabilities[probabilities > 0.5] = 1
        probabilities[probabilities <= 0.5] = 0
        return probabilities.astype(int)

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        ''' Compute target probablities of positive labels '''

        def helper(
            X: pd.DataFrame,
            current_tree_level: dict | float = self.fitted_tree
        ) -> pd.Series:

            if isinstance(current_tree_level, float):
                return pd.Series([current_tree_level for _ in range(len(X))],
                                 index=X.index, dtype=float)

            current_key = list(current_tree_level)[0]
            split_col, split_value, cg = current_key

            X_left = X[X[split_col] <= split_value]
            left_tree = current_tree_level[current_key]['left']
            left_predictions = helper(X_left, left_tree)

            X_right = X[X[split_col] > split_value]
            right_tree = current_tree_level[current_key]['right']
            right_predictions = helper(X_right, right_tree)

            return pd.concat([left_predictions, right_predictions]).sort_index()

        return helper(X)
