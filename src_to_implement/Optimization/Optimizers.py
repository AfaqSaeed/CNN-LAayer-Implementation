import numpy as np

class Sgd:
    def __init__(self, learning_rate: float) -> None:
        """
        Constructor method for Stochastic Gradient Descent (SGD).

        Parameters:
        - learning_rate: The learning rate for updating the weights.
        """
        self.learning_rate = learning_rate

    def calculate_update(self, weight_matrix, gradient_matrix):
        """
        Update the weights using the Stochastic Gradient Descent (SGD) algorithm.

        Parameters:
        - weight_matrix: The current weights of the model.
        - gradient_matrix: The gradient of the loss with respect to the weights.

        Returns:
        - updated_weights: The updated weights after applying the SGD update rule.
        """
        # Applying the SGD update rule to calculate the updated weights
        updated_weights = weight_matrix - (self.learning_rate * gradient_matrix)
        
        return updated_weights
class SgdWithMomentum:
    def __init__(self, learning_rate: float, momentum_rate: float) -> None:
        """
        Constructor method for Stochastic Gradient Descent with Momentum.

        Parameters:
        - learning_rate: The learning rate for updating the weights.
        - momentum_rate: The momentum rate for the momentum term.
        """
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v = 0

    def calculate_update(self, weight_matrix, gradient_matrix):
        """
        Update the weights using the Stochastic Gradient Descent with Momentum algorithm.
        
        Parameters:
        - weight_matrix: The current weights of the model.
        - gradient_matrix: The gradient of the loss with respect to the weights.

        Returns:
        - updated_weights: The updated weights after applying the SGD with momentum update rule.
        """
        # Calculate the momentum term using previous values
        self.v = (self.momentum_rate * self.v) - (self.learning_rate * gradient_matrix)
       
        updated_weights = weight_matrix + self.v

        return updated_weights
class Adam:
    def __init__(self, learning_rate: float, mu: float, rho: float) -> None:
        """
        Constructor method for the Adam optimizer.

        Parameters:
        - learning_rate: The learning rate for updating the weights.
        - mu: The decay rate for the first moment estimate.
        - rho: The decay rate for the second moment estimate.
        """
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.k = 0
        self.v = 0
        self.r = 0

    def calculate_update(self, weight_matrix, gradient_matrix):
        """
        Update the weights using the Adam optimizer algorithm.
        
        Parameters:
        - weight_matrix: The current weights of the model.
        - gradient_matrix: The gradient of the loss with respect to the weights.

        Returns:
        - updated_weights: The updated weights after applying the Adam update rule.
        """
        # Calculate first moment estimate using previous values
        self.v = (self.mu * self.v) + ((1 - self.mu) * gradient_matrix)
        
        # Calculate second moment estimate using previous values
        self.r = (self.rho * self.r) + ((1 - self.rho) * np.square(gradient_matrix))
        
        self.k += 1
        
        # Bias-corrected first moment estimate
        bias_corrected_v = self.v / (1 - np.power(self.mu, self.k))
        
        # Bias-corrected second moment estimate
        bias_corrected_r = self.r / (1 - np.power(self.rho, self.k))
        
        # Applying the Adam update rule to calculate the updated weights
        updated_weights = weight_matrix - ((self.learning_rate * bias_corrected_v) / (np.sqrt(bias_corrected_r) + np.finfo(float).eps))
        
        return updated_weights