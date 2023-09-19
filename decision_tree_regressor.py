"""
Custom decision tree regressor
"""
from typing import Tuple

import numpy as np
import pandas as pd


class MyTreeReg:
    """
    Custom decision tree regressor class
    """
    def __init__(
        self,
        max_depth: int = 5,
        min_samples_split: int = 2,
        max_leafs: int = 20,
        bins: int = None
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.leafs_cnt = 0
        self.bins = bins

    def __str__(self) -> str:
        return (
            f'MyTreeReg class: max_depth={self.max_depth}, '
            f'min_samples_split={self.min_samples_split}, '
            f'max_leafs={self.max_leafs}'
        )

    def _get_feature_importance(
        self,
        left_sample: pd.Series,
        right_sample: pd.Series
    ) -> float:
        node_mse = self._get_mse(pd.concat([left_sample, right_sample]))
        node_mse_gain = self._get_mse_gain(node_mse, left_sample, right_sample)
        node_num = len(left_sample) + len(right_sample)
        return node_num/self.total_samples * node_mse_gain

    def _get_mse(self, target: pd.Series) -> float:
        return ((target - target.mean())**2).mean()

    def _get_mse_gain(
        self,
        current_mse: float,
        left_target: pd.Series,
        right_target: pd.Series
    ) -> float:
        left_n, right_n = len(left_target), len(right_target)
        left_mse, right_mse = self._get_mse(left_target), self._get_mse(right_target)
        return current_mse - left_n/(left_n + right_n) * left_mse - \
            right_n/(left_n + right_n) * right_mse

    def _get_thresholds(self, X: pd.DataFrame) -> dict:
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
        """
        Fit model
        """
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

            split_column, split_value, mse_gain = self.get_best_split(X, y)
            X_left, y_left = X[X[split_column] <= split_value], y[X[split_column] <= split_value]
            X_right, y_right = X[X[split_column] > split_value], y[X[split_column] > split_value]

            fitted_tree = {(split_column, split_value, mse_gain): {}}
            self.fi[split_column] += self._get_feature_importance(y_left, y_right)

            self.leafs_cnt += 2
            fitted_tree[(split_column, split_value, mse_gain)]['left'] = \
                helper(X_left, y_left, depth + 1, 'child')
            fitted_tree[(split_column, split_value, mse_gain)]['right'] = \
                helper(X_right, y_right, depth + 1, 'child')

            return fitted_tree

        self.fi = {col: 0 for col in X.columns}
        self.thresholds = self._get_thresholds(X)
        self.total_samples = X.shape[0]
        self.fitted_tree = helper(X, y)

    def get_best_split(self, X: pd.DataFrame, y: pd.Series) -> Tuple[str, float, float]:
        """
        Iterate through all features and their thresholds to find split with largest mse decrease
        """
        split_col, split_value, mse_gain = 0, 0, 0
        current_mse = self._get_mse(y)

        for col in X.columns:
            for threshold in self.thresholds[col]:
                y_left, y_right = y[X[col] <= threshold], y[X[col] > threshold]
                current_mse_gain = self._get_mse_gain(current_mse, y_left, y_right)

                if current_mse_gain > mse_gain:
                    split_col, split_value, mse_gain = col, threshold, current_mse_gain

        return split_col, split_value, mse_gain

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict target values for a dataset
        """

        def helper(
            X: pd.DataFrame,
            current_tree_level: dict | float = self.fitted_tree
        ) -> pd.Series:

            if isinstance(current_tree_level, float):
                return pd.Series([current_tree_level for _ in range(len(X))],
                                 index=X.index, dtype=float)

            current_key = list(current_tree_level)[0]
            split_col, split_value, mse_gain = current_key

            X_left = X[X[split_col] <= split_value]
            left_tree = current_tree_level[current_key]['left']
            left_predictions = helper(X_left, left_tree)

            X_right = X[X[split_col] > split_value]
            right_tree = current_tree_level[current_key]['right']
            right_predictions = helper(X_right, right_tree)

            return pd.concat([left_predictions, right_predictions]).sort_index()

        return helper(X)
