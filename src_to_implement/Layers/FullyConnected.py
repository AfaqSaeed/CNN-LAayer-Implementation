# Importing necessary libraries
from Layers.Base import BaseLayer
import numpy as np 

# Creating a custom fully connected layer class that inherits from BaseLayer
class FullyConnected(BaseLayer):
    # Getter and setter methods for the optimizer property
    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    # Constructor method for initializing the FullyConnected layer
    def __init__(self, input_size, output_size) -> None:
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.optimizer = None  # Initializing optimizer to None by default
        self.output_size = output_size
        # Initializing weights with random values
        self.weights = np.random.rand(input_size+1, output_size)
        self.bias  = np.ones((output_size,1))
    def initialize(self,weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize((self.input_size+1,self.output_size),self.input_size,self.output_size)
        self.bias = bias_initializer.initialize((self.output_size,1),self.input_size,self.output_size)
        # print(self.bias.shape)
    # Forward pass method for computing the output of the layer
    def forward(self, input_array):
    
        # Appending a column of ones to the input tensor for bias
        input_array_with_ones = np.hstack((input_array, np.ones((input_array.shape[0], 1))))
        
        # Storing the input tensor for later use in the backward pass
        self.input_tensor = input_array_with_ones
        
        # Computing the output tensor using matrix multiplication
        output_array = np.dot(input_array_with_ones, self.weights)
        return output_array

    # Backward pass method for computing gradients and updating weights
    def backward(self, error_tensor):
        # Computing the gradient of the weights using the input tensor and error tensor
        Gradient = np.dot(self.input_tensor.T, error_tensor)
        
        # Computing the error to be propagated to the previous layer
        output_error = np.dot(error_tensor, self.weights[:-1].T)
   
        # Checking if an optimizer is provided and updating the weights accordingly
        if self.optimizer:
            self.weights = self.optimizer.calculate_update(self.weights, Gradient)
        
        # Storing the computed gradient for later use
        self.gradient_weights = Gradient
        
        return output_error
