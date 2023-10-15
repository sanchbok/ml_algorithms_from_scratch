"""
Implementation of agglomerative hierarchical clustering algorithm
"""
from typing import Literal

import numpy as np
import pandas as pd


class MyAgglomerative:
    """
    Agglomerative hierarchical clustering algorithm
    """
    def __init__(
        self,
        n_clusters: int = 3,
        metric: Literal['euclidean', 'chebyshev', 'manhattan', 'cosine'] = 'euclidean'
    ) -> None:
        self.n_clusters = n_clusters
        self.metric = metric

    def __str__(self) -> str:
        return f'MyAgglomerative class: n_clusters={self.n_clusters}'

    @staticmethod
    def compute_pairwise_distances(
        array_1: np.ndarray,
        array_2: np.ndarray,
        metric: str
    ) -> np.ndarray:
        """
        Compute pairwise distances between data samples
        """
        if metric == 'euclidean':
            return np.linalg.norm(array_1[:, None, :] - array_2[None, :, :], axis=-1)

        if metric == 'chebyshev':
            return np.max(np.abs(array_1[:, None, :] - array_2[None, :, :]), axis=-1)

        if metric == 'manhattan':
            return np.sum(np.abs(array_1[:, None, :] - array_2[None, :, :]), axis=-1)

        if metric == 'cosine':
            return 1 - (
                np.sum(array_1[:, None, :] * array_2[None, :, :], axis=-1)/np.multiply(
                    np.linalg.norm(array_1, axis=1).reshape(-1, 1),
                    np.linalg.norm(array_2, axis=1)
                )
            )

        raise ValueError('Unknown metric')

    def fit_predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Fit model and predict target clusters
        """
        data = X.copy(deep=True).values
        # insert unique identifiers of rows
        data = np.insert(data, 0, range(len(data)), axis=1)
        centroid_ids = {}

        while len(data) > self.n_clusters:
            distances = self.compute_pairwise_distances(data[:, 1:], data[:, 1:], self.metric)
            np.fill_diagonal(distances, np.max(distances) + 1)

            # find two closest samples
            first_idx, second_idx = np.unravel_index(
                np.argmin(distances), distances.shape
            )

            first_row_id = data[first_idx][0]
            second_row_id = data[second_idx][0]

            # save centroid's samples
            if first_row_id not in centroid_ids and second_row_id not in centroid_ids:
                centroid_ids[first_row_id] = np.array(
                    [data[first_idx, 1:], data[second_idx, 1:]]
                )

            elif first_row_id in centroid_ids and second_row_id not in centroid_ids:
                centroid_ids[first_row_id] = np.vstack(
                    (centroid_ids[first_row_id], data[second_idx, 1:])
                )

            elif first_row_id not in centroid_ids and second_row_id in centroid_ids:
                centroid_ids[first_row_id] = np.vstack(
                    (data[first_idx, 1:], centroid_ids[second_row_id])
                )
                del centroid_ids[second_row_id]

            else:
                centroid_ids[first_row_id] = np.vstack(
                    (centroid_ids[first_row_id], centroid_ids[second_row_id])
                )
                del centroid_ids[second_row_id]

            # merge closest samples into centroid
            data[first_idx][1:] = centroid_ids[first_row_id].mean(axis=0)
            data = data[data[:, 0] != second_row_id]

        return np.argmin(
            self.compute_pairwise_distances(X.values, data[:, 1:], self.metric), axis=-1
        )
