import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))
        b, w = 0.0, np.zeros(X.shape[1])

        for i in range(epochs):
            y_hat = X @ w + b
            mse = np.mean(np.sum(y_hat - y)**2)

            d_w = 2 * (X.T @ (y_hat - y)) / X.shape[0]
            d_b = 2 * np.sum(y_hat - y) / X.shape[0]

            w -= lr * d_w
            b -= lr * d_b

        return (np.round(w, 5), np.round(b, 5))

