''' Custom KNN classifier class '''
from typing import Literal, Union

import numpy as np
import pandas as pd


class MyKNNClf:
    ''' Custom KNN classifier '''
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
        return f'MyKNNClf class: k={self.k}'

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        ''' Fit knn regressor '''
        self.train_X = X.reset_index(drop=True)
        self.train_y = y.reset_index(drop=True)
        self.train_size = X.shape

    def _get_first_k_minimums(self, distances: np.array) -> Union[np.array, np.array]:
        ''' Return first k minimum indices and their distances'''
        indices = np.arange(len(distances))
        distances = np.array([(dist, idx) for dist, idx in zip(distances, indices)])
        distances = np.array(sorted(distances, key=lambda x: x[0])[:self.k])

        return distances[:, 1], distances[:, 0]

    def _compute_distance(self, test_sample: np.array) -> np.array:
        ''' Return array of distances between test sample and train samples '''
        if self.metric == 'euclidean':
            return np.linalg.norm(self.train_X.values - test_sample, ord=2, axis=1)

        if self.metric == 'chebyshev':
            return np.max(np.abs(self.train_X.values - test_sample), axis=1)

        if self.metric == 'manhattan':
            return np.linalg.norm(self.train_X.values - test_sample, ord=1, axis=1)

        if self.metric == 'cosine':
            train_norms = np.linalg.norm(self.train_X.values, ord=2, axis=1)
            test_norm = np.linalg.norm(test_sample, ord=2)
            return 1 - self.train_X.values.dot(test_sample)/(train_norms * test_norm)

        raise ValueError('Unknown metric name')

    def _get_label(self, min_distances_ids: np.array, distances: np.array) -> int:
        ''' Return label based on nearest neighbours labels '''
        labels = self.train_y[min_distances_ids]

        if self.weight == 'uniform':
            return labels.mode().max()

        if self.weight == 'rank':
            ranks = np.arange(1, len(labels) + 1)
            labels = np.array([[i, j] for i, j in zip(labels, ranks)])

            # case for binary classification
            labels = [
                np.sum(1/labels[labels[:, 0] == 0, 1])/np.sum(1/ranks),
                np.sum(1/labels[labels[:, 0] == 1, 1])/np.sum(1/ranks),
            ]

            return np.argmax(labels)

        if self.weight == 'distance':
            labels = np.array([[i, j] for i, j in zip(labels, distances)])

            labels = [
                np.sum(1/labels[labels[:, 0] == 0, 1])/np.sum(1/distances),
                np.sum(1/labels[labels[:, 0] == 1, 1])/np.sum(1/distances),
            ]

            return np.argmax(labels)

        raise ValueError('Unknown weight name')

    def _get_label_proba(self, min_distances_ids: np.array, distances: np.array) -> float:
        ''' Return probability of positive label based on nearest neighbours labels '''
        labels = self.train_y[min_distances_ids]

        if self.weight == 'uniform':
            return labels.sum()/len(labels)

        if self.weight == 'rank':
            ranks = np.arange(1, len(labels) + 1)
            labels = np.array([[i, j] for i, j in zip(labels, ranks)])

            return np.sum(1/labels[labels[:, 0] == 1, 1])/np.sum(1/ranks)

        if self.weight == 'distance':
            labels = np.array([[i, j] for i, j in zip(labels, distances)])

            return np.sum(1/labels[labels[:, 0] == 1, 1])/np.sum(1/distances)

        raise ValueError('Unknown weight name')

    def predict(self, test_X: pd.DataFrame) -> np.array:
        ''' Return target labels '''
        test_y = np.array([])

        for idx in range(test_X.shape[0]):
            distances = self._compute_distance(test_X.values[idx])
            labels_ids, min_distances = self._get_first_k_minimums(distances)
            test_y = np.append(test_y, self._get_label(labels_ids, min_distances))

        return test_y.astype(int)

    def predict_proba(self, test_X: pd.DataFrame) -> np.array:
        ''' Return probabilities of positive labels '''
        test_y = np.array([])

        for idx in range(test_X.shape[0]):
            distances = self._compute_distance(test_X.values[idx])
            labels_ids, min_distances = self._get_first_k_minimums(distances)
            test_y = np.append(test_y, self._get_label_proba(labels_ids, min_distances))

        return test_y
