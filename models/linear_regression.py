''' Custom linear regression class '''
from typing import Callable

import random

import numpy as np
import pandas as pd


class MyLineReg:
    ''' Custom linear regression '''
    def __init__(
        self,
        n_iter: int = 100,
        learning_rate: float | Callable[[int], float] = 0.1,
        metric: str = None,
        reg: str = None,
        l1_coef: float = 0,
        l2_coef: float = 0,
        sgd_sample: int | float = None,
        random_state: int = 42
    ) -> None:

        if reg and not (l1_coef or l2_coef):
            raise ValueError("l1_coef and l2_coef can't be None if reg is True")

        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = np.array([])
        self.metric = metric
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.__best_score = None
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    def __str__(self) -> str:
        return f'MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'

    def get_mse_loss(self, y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
        ''' Compute MSE loss '''
        loss = ((y_true - y_pred)**2).mean()

        if self.reg:

            if self.reg == 'l1':
                return loss + self.l1_coef * np.abs(self.weights).sum()

            if self.reg == 'l2':
                return loss + self.l2_coef * (self.weights ** 2).sum()

            return loss + self.l1_coef * np.abs(self.weights).sum() \
                + self.l2_coef * (self.weights ** 2).sum()

        return loss

    def get_gradient(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        X: pd.DataFrame
    ) -> pd.Series:
        """
        Compute gradient vector 
        """
        gradient = (y_pred - y_true).dot(X) * 2/X.shape[0]

        if self.reg:
            weights_grad = self.weights.copy()

            if self.reg == 'l1':
                weights_grad[weights_grad < 0] = -1
                weights_grad[weights_grad > 0] = 1
                return gradient + self.l1_coef * weights_grad

            if self.reg == 'l2':
                return gradient + 2 * self.l2_coef * weights_grad

            weights_grad[weights_grad < 0] = -1
            weights_grad[weights_grad > 0] = 1
            return gradient + self.l1_coef * weights_grad + 2 * self.l2_coef * self.weights

        return (y_pred - y_true).dot(X) * 2/X.shape[0]

    def get_metric(self, y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
        ''' Compute metric '''
        if self.metric == 'mae':
            return (y_true - y_pred).abs().mean()

        if self.metric == 'mse':
            return ((y_true - y_pred)**2).mean()

        if self.metric == 'rmse':
            return ((y_true - y_pred)**2).mean()**0.5

        if self.metric == 'mape':
            return ((y_true - y_pred)/y_true).abs().mean() * 100

        if self.metric == 'r2':
            return 1 - ((y_true - y_pred)**2).sum()/((y_true - y_true.mean())**2).sum()

        raise ValueError('Unknown metric')

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: bool | int = False) -> None:
        ''' Optimize weights of regression '''
        random.seed(self.random_state)

        # copy data
        copy_X = X.reset_index(drop=True)
        copy_y = y.reset_index(drop=True)

        # add consant column
        copy_X.insert(0, 'const', 1)

        # initialize weights
        self.weights = np.ones(copy_X.shape[1])

        for iter in range(1, self.n_iter + 1):

            # sample random batch for SGD
            if isinstance(self.sgd_sample, int):
                sample_rows_idx = random.sample(range(X.shape[0]), self.sgd_sample)

            elif isinstance(self.sgd_sample, float):
                sample_rows_idx = random.sample(
                    range(X.shape[0]), int(X.shape[0] * self.sgd_sample)
                )

            else:
                sample_rows_idx = list(range(X.shape[0]))

            current_X = copy_X.loc[sample_rows_idx]
            current_y = copy_y[sample_rows_idx]

            # make predictions
            y_pred = current_X.dot(self.weights)

            # print intermediate result
            if verbose:
                if iter % verbose == 0 and self.metric:
                    print(f'iter {iter}', f'loss: {self.get_mse_loss(y, copy_X.dot(self.weights))}',
                          f'{self.metric}: {self.get_metric(y, copy_X.dot(self.weights))}', 
                          sep=' | ')

                elif iter % verbose == 0:
                    print(f'iter | loss: {self.get_mse_loss(y, copy_X.dot(self.weights))}')

            # compute gradient
            gradient = self.get_gradient(current_y, y_pred, current_X)

            # optimize weights
            if isinstance(self.learning_rate, Callable):
                self.weights -= self.learning_rate(iter) * gradient
            else:
                self.weights -= self.learning_rate * gradient

        # compute final metric score
        if self.metric:
            self.__best_score = self.get_metric(y, copy_X.dot(self.weights))

    def get_coef(self) -> np.array:
        ''' Return weights values '''
        return np.array(self.weights)[1:]

    def predict(self, X: pd.DataFrame) -> pd.Series:
        ''' Return predictions '''
        copy_X = X.copy(deep=True)
        copy_X.insert(0, 'const', 1)

        return copy_X.dot(self.weights)

    def get_best_score(self) -> float:
        ''' Return latest metric value '''
        return self.__best_score
