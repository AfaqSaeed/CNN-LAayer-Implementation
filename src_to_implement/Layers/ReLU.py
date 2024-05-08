# Importing necessary libraries
from Layers.Base import BaseLayer
import numpy as np

# Defining a class for the ReLU activation layer, inheriting from BaseLayer
class ReLU(BaseLayer):
    def __init__(self):
        # Calling the constructor of the parent class (BaseLayer)
        super().__init__()

    # Forward pass method for ReLU activation
    def forward(self, Matrix):
        # Storing the input matrix for later use in the backward pass
        self.Matrix = Matrix
        # Applying ReLU activation element-wise on the input matrix
        return np.maximum(0, Matrix)

    # Backward pass method for ReLU activation
    def backward(self, error_Matrix):
        # Creating a copy of the error matrix to avoid modifying the original
        return_Matrix = error_Matrix.copy()
        # Setting the gradient to 0 for the elements where the corresponding
        # input in the forward pass was less than 0
        return_Matrix[self.Matrix < 0] = 0
        # Returning the computed gradient for the backward pass
        return return_Matrix
