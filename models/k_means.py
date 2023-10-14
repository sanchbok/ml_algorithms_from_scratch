"""
Custom K-Means implementation
"""
from typing import List, Tuple

import numpy as np
import pandas as pd


class MyKMeans:
    """
    K-Means model
    """
    def __init__(
        self,
        n_clusters: int = 3,
        max_iter: int = 10,
        n_init: int = 3,
        random_state: int = 42
    ) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state

    def __str__(self) -> str:
        return (
            f'MyKMeans class: n_clusters={self.n_clusters}, '
            f'max_iter={self.max_iter}, n_init={self.n_init}, random_state={self.random_state}'
        )

    def _get_best_centroids(
        self,
        cluster_centroids: List[np.ndarray],
        X: pd.DataFrame
    ) -> Tuple[np.ndarray, float]:

        best_centroids = np.zeros((self.n_clusters, X.shape[1]))
        best_wcss = 1e10

        for centroids in cluster_centroids:
            current_clusters = np.argmin(
                self.euclidean_distance(X, centroids), axis=-1
            )

            wcss = 0
            for i, centroid in enumerate(centroids):
                wcss += (
                    np.linalg.norm(
                        X.values[current_clusters == i] - centroid, axis=1
                    )**2
                ).sum()

            if wcss < best_wcss:
                best_wcss = wcss
                best_centroids = centroids

        return best_centroids, best_wcss

    @staticmethod
    def euclidean_distance(X: pd.DataFrame, centroids: np.ndarray) -> np.array:
        """
        Compute euclidean distance between samples and cluster centroids
        """
        return np.linalg.norm(
            X.values[:, None, :] - centroids[None, :, :], axis=-1
        )

    def fit(self, X: pd.DataFrame) -> None:
        """
        Fit model
        """
        np.random.seed(self.random_state)
        cluster_centroids = []

        for _ in range(self.n_init):
            # genereate initial centroids
            centroids = np.random.uniform(
                X.min(axis=0), X.max(axis=0), size=(self.n_clusters, X.shape[1])
            )

            # optimize centroid coordinates
            for _ in range(self.max_iter):
                current_clusters = np.argmin(
                    self.euclidean_distance(X, centroids), axis=-1
                )

                new_centroids = centroids.copy()

                for i in range(self.n_clusters):
                    if i in current_clusters:
                        new_centroids[i] = X.values[current_clusters == i].mean(axis=0)

                if np.array_equal(new_centroids, centroids):
                    break

                centroids = new_centroids.copy()

            # save found centroids
            cluster_centroids.append(centroids)

        # find best combination of centroids
        self.cluster_centers_, self.inertia_ = self._get_best_centroids(cluster_centroids, X)

    def predict(self, X: pd.DataFrame) -> np.array:
        """
        Predict cluster labels for test samples
        """
        return np.argmin(
            self.euclidean_distance(X, self.cluster_centers_), axis=-1
        )
