class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        for _ in range(iterations):
            df = 2*init# Derivative:         f'(x) = 2x
            init = init - learning_rate*df # Update rule:        x = x - learning_rate * f'(x)
        init = round(init, 5) # Round final answer to 5 decimal places

        return init
        
# time: O(iterations)
# space: O(1) 