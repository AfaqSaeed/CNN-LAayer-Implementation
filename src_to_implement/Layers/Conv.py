import numpy as np
from Layers.Base import BaseLayer
from numpy.lib.stride_tricks import as_strided
import math
import scipy
class Conv(BaseLayer):
    def __init__(self,stride_shape,convolution_shape,num_kernels) -> None:
        super().__init__()
        self.trainable = True
        
        self.num_kernels = num_kernels
        self.bias = np.random.uniform(0,1,(self.num_kernels))

        if len(stride_shape)==1:
            self.stride_shape = (1,stride_shape[0])
        else:
            self.stride_shape = stride_shape
        if len(convolution_shape)==2:
            self.convolution_shape = (convolution_shape[0],1,convolution_shape[1])
            # print("COnvolution Shape : ",self.convolution_shape)
        else:
            self.convolution_shape=convolution_shape
        self.weights = np.random.uniform(0,1,(self.num_kernels,self.convolution_shape[0],self.convolution_shape[1],self.convolution_shape[2]))
        
    def initialize(self,weights_initializer,bias_initializer):
        weights_shape = (self.num_kernels,self.convolution_shape[0],self.convolution_shape[1],self.convolution_shape[2])
        # print("Weights",weights_shape)
        fan_out = np.prod(np.array([self.num_kernels,self.convolution_shape[1],self.convolution_shape[2]]))
        self.weights = weights_initializer.initialize(weights_shape,np.prod(self.convolution_shape),fan_out)
        self.bias = bias_initializer.initialize((self.num_kernels),self.convolution_shape[0],self.num_kernels)


    def forward(self,input_tensor):
        self.input_tensor=input_tensor
        if len(input_tensor.shape)==3:
            
            input_tensor = np.expand_dims(input_tensor,2)
            batch_size, in_channels, in_width, in_height = input_tensor.shape

        else:
            
            
            
            batch_size, in_channels, in_width, in_height = input_tensor.shape
       
        out_width =  math.ceil(in_width / self.stride_shape[0])  
        out_height = math.ceil((in_height / self.stride_shape[1]))
        outputs = np.zeros((batch_size,self.num_kernels, out_width, out_height), dtype=np.float32)
        # print("In_width : ",input_tensor.shape[2],"In_width_padded : ",in_width,"Out_width : ",out_width,"Stride",self.stride_shape,"Kernel size: ",self.convolution_shape[1:])
        # print("In_height : ",input_tensor.shape[3] if len(input_tensor.shape)==4 else 1 ,"In_height_padded : ",in_height,"Out_height : ",out_height,"Stride",self.stride_shape,"Kernel size: ",self.convolution_shape[1:])
        # print("Outputs Shape",outputs.shape)
        # print("Num kernels",self.num_kernels)
        for n in range(0,outputs.shape[0]):
            for cout in range(0,outputs.shape[1]):
                weights_slice =self.weights[cout,:,:,:]
                input_slice = input_tensor[n,:,:,:] 
                # if n==1:
                    # print(input_slice,weights_slice)
                convolve = scipy.signal.correlate(input_slice,weights_slice,mode= "same")
                # print(convolve.shape)

            
                outputs[n,cout,:,:] = np.sum(convolve,axis=0)[::self.stride_shape[0], ::self.stride_shape[1]]+self.bias[cout]
        

        self.input_tensor = input_tensor
        if outputs.shape[2]==1:
            outputs=outputs.squeeze(axis=2)

        return outputs

    def backward(self, dL_dout):
        # Get dimensions
        # conv_input = self.input_tensor 
        # n_filters, d_filter, h_filter, w_filter = self.weights.shape
        # n_x, d_x, h_x, w_x = conv_input.shape
        # n_out, d_out, h_out, w_out = dL_dout.shape

        # # Initialize gradients
        # dL_dconv = np.zeros(conv_input.shape)
        # dL_dfilter = np.zeros(self.weights.shape)

        # # Pad input
        # conv_input_padded = np.pad(conv_input, ((0, 0), (0, 0), (0, 0), (0,0)), mode='constant')

        # # Loop over filters
        # for f in range(n_filters):
        #     # Loop over output dimensions
        #     for i in range(h_out):
        #         for j in range(w_out):
        #             # Compute gradients
        #             dL_dconv[:, :, i*self.stride_shape[0]:i*self.stride_shape[0]+h_filter, j*self.stride_shape[1]:j*self.stride_shape[1]+w_filter] += \
        #                 self.weights[f, :, :, :] * dL_dout[f, :, i, j]
        #             dL_dfilter[f, :, :, :] += \
        #                 conv_input_padded[:, :, i*self.stride_shape[0]:i*self.stride_shape[0]+h_filter, j*self.stride_shape[1]:j*self.stride_shape[1]+w_filter] * dL_dout[f, :, i, j]

        return dL_dout
