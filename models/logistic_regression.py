''' Custom logistic regression class '''
from typing import Callable, Literal

import random

import numpy as np
import pandas as pd


class MyLogReg:
    ''' Custom logistic regression '''
    def __init__(
        self,
        n_iter: int = 10,
        learning_rate: float | Callable[[int], float] = 0.1,
        metric: Literal['accuracy', 'precision', 'recall', 'f1', 'roc_auc'] = None,
        reg: Literal['l1', 'l2', 'elasticnet'] = None,
        l1_coef: float = 0,
        l2_coef: float = 0,
        sgd_sample: int | float = None,
        random_state: int = 42
    ) -> None:
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.metric = metric
        self.weights = np.array([])
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    def __str__(self) -> str:
        return f'MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'

    def get_log_loss(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        ''' Compute log loss '''
        eps = 1e-15
        loss = - np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))

        if self.reg:

            if self.reg == 'elasticnet':
                return loss + self.l1_coef * np.abs(self.weights).sum() + \
                    self.l2_coef * (self.weights**2).sum()

            if self.reg == 'l1':
                return loss + self.l1_coef * np.abs(self.weights).sum()

            return loss + self.l2_coef * (self.weights**2).sum()

        return loss

    def get_gradient(self, y_true: pd.Series, y_pred: pd.Series, X: pd.DataFrame) -> np.array:
        ''' Compute gradient of log loss '''
        gradient = (y_pred - y_true).dot(X)/X.shape[0]
        if self.reg:
            weights_grad = self.weights.copy()

            if self.reg == 'elasticnet':
                weights_grad[weights_grad > 0] = 1
                weights_grad[weights_grad < 0] = -1
                return gradient + self.l1_coef * weights_grad + self.l2_coef * 2 * self.weights

            if self.reg == 'l1':
                weights_grad[weights_grad > 0] = 1
                weights_grad[weights_grad < 0] = -1
                return gradient + self.l1_coef * weights_grad

            return gradient + self.l2_coef * 2 * self.weights

        return gradient

    def get_metric(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        ''' Compute metric '''
        if self.metric == 'roc_auc':
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

        else:
            pred_labels = y_pred.copy(deep=True)
            pred_labels[pred_labels > 0.5] = 1
            pred_labels[pred_labels <= 0.5] = 0

            if self.metric == 'accuracy':
                return sum(pred_labels == y_true)/len(y_true)

            elif self.metric == 'precision':
                return sum(pred_labels[pred_labels == y_true] == 1)/sum(pred_labels == 1)

            elif self.metric == 'recall':
                return sum(pred_labels[pred_labels == y_true] == 1)/sum(y_true == 1)

            elif self.metric == 'f1':
                precision = sum(pred_labels[pred_labels == y_true] == 1)/sum(pred_labels == 1)
                recall = sum(pred_labels[pred_labels == y_true] == 1)/sum(y_true == 1)
                return (2 * precision * recall)/(precision + recall)

            raise ValueError('Unknown metric name')

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: bool | int = False) -> None:
        ''' Optimize weights of regression '''
        random.seed(self.random_state)

        copy_X = X.reset_index(drop=True)
        copy_y = y.reset_index(drop=True)

        # add constant column
        copy_X.insert(0, 'const', 1)

        # initialize weights
        self.weights = np.ones(copy_X.shape[1])

        for iter in range(1, self.n_iter + 1):

            # sample random batch for SGD
            if isinstance(self.sgd_sample, int):
                sample_rows_idx = random.sample(range(X.shape[0]), self.sgd_sample)

            elif isinstance(self.sgd_sample, float):
                sample_rows_idx = random.sample(
                    range(X.shape[0]), int(self.sgd_sample * X.shape[0])
                )

            else:
                sample_rows_idx = list(range(X.shape[0]))

            current_X = copy_X.loc[sample_rows_idx]
            current_y = copy_y[sample_rows_idx]

            logits = current_X.dot(self.weights)
            y_pred = pd.Series(1/(1 + np.e**(-logits)))

            loss = self.get_log_loss(current_y, y_pred)
            if verbose:
                if iter % verbose == 0:
                    if self.metric:
                        print(f'{iter} | loss: {round(loss, 2)} | ',
                              f'{self.metric}: {self.get_metric(current_y, y_pred)}')
                    else:
                        print(f'{iter} | loss: {round(loss, 2)}')

            grad = self.get_gradient(current_y, y_pred, current_X)
            if isinstance(self.learning_rate, Callable):
                self.weights -= self.learning_rate(iter) * grad
            else:
                self.weights -= self.learning_rate * grad

        if self.metric:
            logits = copy_X.dot(self.weights)
            y_pred = pd.Series(1/(1 + np.e**(-logits)))
            self._best_score = self.get_metric(y, y_pred)

    def get_best_score(self) -> float:
        ''' Return best metric score '''
        return self._best_score

    def get_coef(self) -> np.array:
        ''' Return regression coefficients '''
        return np.array(self.weights)[1:]

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        ''' Return computed target probabilities '''
        copy_X = X.copy(deep=True)
        copy_X.insert(0, 'const', 1)

        logits = copy_X.dot(self.weights)
        proba = pd.Series(1/(1 + np.e**(-logits)))
        return proba

    def predict(self, X: pd.DataFrame) -> pd.Series:
        ''' Return computed targets '''
        proba = self.predict_proba(X)
        proba[proba > 0.5] = 1
        proba[proba <= 0.5] = 0
        return proba.astype(int)
