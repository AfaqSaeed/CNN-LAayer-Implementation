# Importing necessary libraries
from Layers.Base import BaseLayer
import numpy as np

# Defining a class for the Softmax activation layer, inheriting from BaseLayer
class SoftMax(BaseLayer):
    def __init__(self):
        # Calling the constructor of the parent class (BaseLayer)
        super().__init__()

    # Forward pass method for Softmax activation
    def forward(self, input_tensor):
        # Finding the maximum value along each row
        max_values = np.max(input_tensor.copy(), axis=1, keepdims=True)
        
        # Calculating the exponential of each element after adjusting for the maximum value
        exponential = np.exp(input_tensor - max_values)
    
        # Calculating the sum of exponentials along each row
        sum_exp = np.sum(exponential, axis=1, keepdims=True)
        
        # Computing the final Softmax predictions by dividing each exponential by the sum
        predictions = np.divide(exponential, sum_exp)
        
        # Storing the predictions for later use in the backward pass
        self.predictions = predictions
    
        return predictions
    
    # Backward pass method for Softmax activation
    def backward(self, error_tensor):
        # Element-wise multiplication of the error tensor with the stored Softmax predictions
        multiplication = np.multiply(error_tensor, self.predictions)
        
        # Summing the multiplied values along each row
        summation = np.sum(multiplication, axis=1, keepdims=True)
        
        # Calculating the difference between the error tensor and the summed values
        error_differences = error_tensor - summation
        
        # Final computation of the backward pass by multiplying with the stored Softmax predictions
        error_backward = np.multiply(error_differences, self.predictions)
        
        return error_backward
