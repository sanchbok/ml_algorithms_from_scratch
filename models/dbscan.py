"""
Implementation of DBSCAN 
"""
from typing import Literal

import numpy as np
import pandas as pd


class MyDBSCAN:
    """
    Implementation of DBSCAN 
    """
    def __init__(
        self,
        eps: int = 3,
        min_samples: int = 3,
        metric: Literal['euclidean', 'chebyshev', 'manhattan', 'cosine'] = 'euclidean'
    ) -> None:
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric

    def __str__(self) -> str:
        return f'MyDBSCAN class: eps={self.eps}, min_samples={self.min_samples}'

    @ staticmethod
    def compute_distance(vector: np.ndarray, matrix: np.ndarray, metric: str) -> np.array:
        """
        Compute pairwise distances between vector and matrix
        """
        if metric == 'euclidean':
            return np.linalg.norm(vector - matrix, axis=1)

        if metric == 'chebyshev':
            return np.max(np.abs(vector - matrix), axis=1)

        if metric == 'manhattan':
            return np.sum(np.abs(vector - matrix), axis=1)

        if metric == 'cosine':
            return 1 - np.sum(vector * matrix, axis=1)/ \
                np.sqrt(np.sum(vector**2)*np.sum(matrix**2, axis=1))

        raise ValueError('Unknown metric')

    def fit_predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Fit model 
        """
        data = X.copy(deep=True).values
        labels = np.zeros(data.shape[0])

        cluster_id = 1
        for i, label in enumerate(labels):
            if label != 0:
                continue

            distances = self.compute_distance(data[i], data, self.metric)
            neighbours = np.where(distances <= self.eps)[0]

            if len(neighbours) > self.min_samples:
                labels[i] = cluster_id
                k = 0
                while k < len(neighbours):
                    neighbour = neighbours[k]
                    if labels[neighbour] == 0:
                        labels[neighbour] = cluster_id

                        next_distances = self.compute_distance(data[neighbour], data, self.metric)
                        next_neighbours = np.where(next_distances <= self.eps)[0]

                        if len(next_neighbours) > self.min_samples:
                            neighbours = np.append(neighbours, next_neighbours)

                    k += 1

            cluster_id += 1

        return labels
