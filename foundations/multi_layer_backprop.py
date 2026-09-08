import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        x, W1, b1, W2, b2, y_true = np.array(x), np.array(W1), np.array(b1),\
         np.array(W2), np.array(b2), np.array(y_true)

        z1 = np.dot(W1, x) + b1  #first linear. shape (hidden_size,) = (2,)
        a1 = np.maximum(0, z1) #first activation. shape (hidden_size,) = (2,)
        
        z2 = np.dot(W2, a1) + b2  #second linear. shape (output_size,) = (1,)
        loss = np.mean((z2 - y_true)**2) #mse loss. float

        dz2 = 2*(z2 - y_true)/len(z2)  #deriv wrt z2. shape (1,)
        dW2 = np.outer(dz2, a1) #deriv wrt W2. shape (output_shape, hidden_size) = (1,2)
        db2 = dz2 #deriv wrt b2. shape (output_shape,) = (1,)

        da1 = dz2 @ W2 #deriv wrt a1. shape (hidden_size,) = (2,)
        dz1 = da1 * (z1 > 0) #deriv wrt z1. shape (hidden_size,) = (2,)

        dW1 = np.outer(dz1, x) #deriv wrt W1. shape (hidden_size, input_size) = (2,2)
        db1 = dz1 #deriv wrt b1. shape (hidden_size,) = (2,)
        
        return {'loss': np.round(float(loss), 4), 'dW1': np.round(dW1.tolist(), 4),\
        'db1': np.round(db1.tolist(), 4),\
        'dW2': np.round(dW2.tolist(), 4),'db2': np.round(db2.tolist(), 4)}
