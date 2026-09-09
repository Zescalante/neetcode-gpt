import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer

        for i in range(len(weights)):   #iterate through layers
            z = x @ weights[i] + biases[i]  #linear layer. shape = (num_layer_nodes,)
            if i < len(weights) - 1:    #if not last layer, apply relu
                x = np.maximum(0, z)
            else:   #else no activation for last layer
                x = z
        
        return np.round(x, 5)

# time: O(n); n = number of els in the input
# space: O(n)
