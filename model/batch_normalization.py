import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        x, gamma, beta, running_mean, running_var = np.array(x), np.array(gamma),np.array(beta),np.array(running_mean),np.array(running_var)

        if training:
            mu_b = np.mean(x, axis=0)
            var_b = np.mean((x - mu_b)**2, axis=0)
            x_hat = (x - mu_b) / np.sqrt(var_b + eps)

            running_mean = (1.0 - momentum) * running_mean + momentum * mu_b
            running_var = (1.0 - momentum) * running_var + momentum * var_b
        else:
            x_hat = (x - running_mean) / np.sqrt(running_var + eps)

        y = gamma * x_hat + beta
        return (list(np.round(y, 4)), list(np.round(running_mean, 4)), list(np.round(running_var, 4)))
