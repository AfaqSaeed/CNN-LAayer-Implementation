import copy

class NeuralNetwork:
    
    def __init__(self,optimizer,weights_initializer,bias_initializer) -> None:
        ### initialize all the variables asked in the exercise fdndsk
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        
        self.optimizer = optimizer
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.loss = []

    def append_layer(self,layer):
        if layer.trainable: ######### If layer is trainable then just make a copy of the array  
        
            layer.optimizer = copy.deepcopy(self.optimizer)
            layer.initialize(self.weights_initializer,self.bias_initializer)
    
        self.layers.append(layer) ### at every call append the layer to the main list of layers 
            
    def forward(self): ########## Do one forward pass by iterating over all the layers 
        in_Array,Label=self.data_layer.next()
        self.Label = Label
        
        first_layer = True ## Varialble to detect whether its first layer or not 
        
        for L in self.layers:
            if first_layer:
                output_tensor = L.forward(in_Array)
                first_layer = False
            else:
                output_tensor = L.forward(output_tensor)
        
        self.prediction = output_tensor
        loss =self.loss_layer.forward(output_tensor,Label) 
       
        return loss
    def backward(self):
    
        gradient = self.loss_layer.backward(self.Label) 

        for l in reversed(self.layers): ######### Iterate through the array in reverse order please 

            ###Gradient backpropagation 
            gradient = l.backward(gradient)

        ###

    def train(self,loops):
    
        for _ in range(0,loops): ### Just iterate over the loops value and call forward and backward methods while appending loss 
            loss = self.forward()
            self.loss.append(loss)
            self.backward()
    
    def test(self,in_Array): #### Call the forward method once but can take input this time 
    
        first_layer = True
        for l in self.layers:
            if first_layer:
                out_Array = l.forward(in_Array)
                first_layer = False
            else:
                out_Array = l.forward(out_Array)
        
        return out_Array
