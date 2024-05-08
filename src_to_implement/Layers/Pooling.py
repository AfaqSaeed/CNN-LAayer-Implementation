from Layers.Base import BaseLayer
import math
import numpy as np


class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape


    def forward(self,input_matrix):

        batch_size, in_channels, in_width, in_height = input_matrix.shape
        self.input = input_matrix
        out_width =  int((in_width - self.pooling_shape[0]) / (self.stride_shape[0])) + 1  
        out_height = int((in_height - self.pooling_shape[1]) / (self.stride_shape[1])) + 1
        
        self.highest_num_locations = {}  # Keep track of the indices of highest values for each cropped region with dictionary

        
        out_image_dim = (in_channels, out_width,out_height)

        # Create a result matrix with zeros
        outputs = np.zeros((batch_size, *out_image_dim))

        # Loop through every batch in the input matrix
        for batch_id in range(batch_size):
            # Loop through every channel in the input matrix
            for in_channel in range(out_image_dim[0]):
                # Loop through every x_dimension in the input matrix
                for output_x in range(out_image_dim[1]):
                # Loop through every y_dimension in the input matrix
                    for output_y in range(out_image_dim[2]):
                        # calculate input_dimensions with stride for clipping
                        input_x = self.stride_shape[0] * output_x
                        input_y = self.stride_shape[1] * output_y

                        # Clip the region of interest
                        a = input_matrix[
                                     batch_id,
                                     in_channel,
                                     input_x:input_x + self.pooling_shape[0],
                                     input_y:input_y + self.pooling_shape[1]]
                        # Create a key specifiying the starting position of the sliding kernel 
                        keydim = (batch_id, in_channel, output_x, output_y)
                        # By default argmax flattens array and returns indices according to flattened array so convert back to 2d coords we use unravel indexes
                        highest_num_location = np.unravel_index(a.argmax(), a.shape)
                        # At this key assign values to be used in backward pass
                        self.highest_num_locations[keydim] = (highest_num_location[0] + input_x, highest_num_location[1] + input_y)

                        # Place highest position number in the the correct position in the output matrix
                        outputs[keydim] = a[highest_num_location]
        return outputs


    def backward(self, error_matrix):
        #Create a zero filled matrix of same shape as input 
        zero_filled_error_matrix = np.zeros(self.input.shape)
        # iterate through each array to get keys and their corresponding max values 
        # and using assign the values given at the exact position we took the highest number from
        for shapepos in self.highest_num_locations.keys():
            highest_location = self.highest_num_locations[shapepos]
            shape = (shapepos[0], shapepos[1], *highest_location)
            zero_filled_error_matrix[shape] = zero_filled_error_matrix[shape] + error_matrix[shapepos]
        
        return zero_filled_error_matrix