import numpy as np
from Layers.Base import BaseLayer
class Flatten(BaseLayer):
    def __init__(self) -> None:
       super().__init__() 
    def forward(self,input):
        
        self.input_tensor_shape = input.shape 
        flattendim = (input.shape[0],np.prod(input.shape[1:] ))

        out = np.reshape(input.copy(),flattendim)
        return out 
    def backward(self,input):
        out = np.reshape(input,self.input_tensor_shape)
        return out 
        
