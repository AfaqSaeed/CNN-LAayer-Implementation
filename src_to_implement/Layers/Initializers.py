import numpy as np 
class Constant:
    def __init__(self,init_val = 0.1):
        self.init_val = init_val            
    def initialize(self,weights_shape,fan_in,fan_out):
        weights = np.full((weights_shape),self.init_val)
        return weights

class UniformRandom:
    def __init__(self):
        pass            
    def initialize(self,weights_shape,fan_in,fan_out):
        weights = np.random.uniform(0,1,weights_shape)
        return weights
class Xavier:
    def __init__(self):
        pass            
    def initialize(self,weights_shape,fan_in,fan_out):
        mean = 0
        sigma = np.sqrt(2/(fan_in+fan_out))
        weights = np.random.normal(mean,sigma,weights_shape)
        return weights
class He:
    def __init__(self):
        pass            
    def initialize(self,weights_shape,fan_in,fan_out):
        mean = 0
        sigma = np.sqrt(2/(fan_in))
        weights = np.random.normal(mean,sigma,weights_shape)
        return weights