''' Custom KNN regressor class '''
from typing import Literal, Union

import numpy as np
import pandas as pd


class MyKNNReg:
    ''' Custom KNN regressor '''
    def __init__(
        self,
        k: int = 3,
        metric: Literal['euclidean', 'chebyshev', 'manhattan', 'cosine'] = 'euclidean',
        weight: Literal['uniform', 'rank', 'distance'] = 'uniform'
    ) -> None:
        self.k = k
        self.train_size = 0
        self.metric = metric
        self.weight = weight

    def __str__(self) -> str:
        return f'MyKNNReg class: k={self.k}'

    def _compute_distance(self, test_obs) -> np.array:
        ''' Compute distance between test observation and train samples '''
        if self.metric == 'euclidean':
            return np.linalg.norm(self.X_train.values - test_obs, ord=2, axis=1)

        if self.metric == 'chebyshev':
            return np.max(np.abs(self.X_train.values - test_obs), axis=1)

        if self.metric == 'manhattan':
            return np.linalg.norm(self.X_train.values - test_obs, ord=1, axis=1)

        if self.metric == 'cosine':
            train_norms = np.linalg.norm(self.X_train.values, ord=2, axis=1)
            test_norm = np.linalg.norm(test_obs, ord=2)
            return 1 - self.X_train.values.dot(test_obs)/(train_norms * test_norm)

        raise ValueError('Unknown metric name')

    def _get_nearest_neighbours(self, distances) -> Union[np.array, np.array]:
        ''' Find first k nearest observations train ids and distances to them '''
        indices = np.arange(0, len(distances))
        nearest_neighbours = np.array(
            sorted(
                zip(indices, distances), key=lambda x: x[1]
            )[:self.k]
        )
        return nearest_neighbours[:, 0], nearest_neighbours[:, 1]

    def _get_target_value(self, nn_ids: np.array, nn_distances: np.array) -> float:
        ''' Compute target value '''
        target_values = self.y_train[nn_ids]

        if self.weight == 'uniform':
            return np.mean(target_values)

        if self.weight == 'rank':
            ranks = np.arange(1, len(nn_ids) + 1)
            return np.sum(
                target_values * (1/ranks)/np.sum(1/ranks)
            )

        if self.weight == 'distance':
            return np.sum(
                target_values * (1/nn_distances)/np.sum(1/nn_distances)
            )

        raise ValueError('Unknown weight name')

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        ''' Remember train data '''
        self.X_train = X.reset_index(drop=True)
        self.y_train = y.reset_index(drop=True)
        self.train_size = X.shape

    def predict(self, test_X: pd.DataFrame) -> np.array:
        ''' Predict target values '''
        targets = np.array([])
        for test_obs in test_X.values:
            distances = self._compute_distance(test_obs)
            nn_ids, nn_distances = self._get_nearest_neighbours(distances)
            targets = np.append(targets, self._get_target_value(nn_ids, nn_distances))
        return targets
