from multiprocessing import pool
import numpy as np
from nn.functions import activations

class Layer(object):
    def __init__(self, _activations : str):
        self.params = []
        
        self.input_shape = None

        self.previous_layer = None
        self.next_layer = None

        self.input = None
        self.output = None
        
        self.accumulated_delta = None
        self.loss_delta = None

        self.activation_function = activations.ActivationFunction(_activations)

    def feed_forward(self):
        raise NotImplementedError
    
    def get_activate(self, data):
        return self.activation_function.get_activate(data)
    
    def get_activate_diff(self, data):
        return self.activation_function.get_activate_diff(data)
    
    def backpropagation(self):
        raise NotImplementedError

    def get_accumulated_delta(self):
        if self.next_layer is not None:
            return self.next_layer.accumulated_delta
        else:
            return self.loss_delta
        
    def get_input(self):

        if self.previous_layer is not None:
            return np.dot(self.weight, self.previous_layer.output) + self.bias
        else:
            return np.array(self.input,dtype=np.float128)

    def clear_deltas(self):
        pass

    def update(self, learning_rate):
        pass
    
    def connect(self, layer):
        self.previous_layer = layer
        layer.next_layer = self
        

    
    
class DenseLayer(Layer):
    def __init__(self,dim=None,activations = None):

        super(DenseLayer, self).__init__(activations)
        self.dim = dim
        
        self.weight = None
        self.bias = None
        self.params = [self.weight, self.bias]

        self.delta_w = None
        self.delta_b = None

    def feed_forward(self):
        self.input = self.get_input()
        self.output = self.activation_function.get_activate(self.input)
    def backpropagation(self):
        data = self.get_input()
        accumulated_delta = self.get_accumulated_delta()
        activate_diff = self.get_activate_diff(data)

        if self.next_layer is None and (accumulated_delta.shape != activate_diff.shape) :
            delta = np.dot(activate_diff.transpose(), accumulated_delta)
        else:
            delta = accumulated_delta * activate_diff
        
        self.delta_b += delta
        self.delta_w += np.dot(delta, self.previous_layer.output.transpose())
        self.accumulated_delta = np.dot(self.weight.transpose(), delta)


    def update(self, learning_rate):
        self.weight -= learning_rate * self.delta_w
        self.bias -= learning_rate * self.delta_b

    def clear_deltas(self):
        self.delta_w = np.zeros(self.weight.shape)
        self.delta_b = np.zeros(self.bias.shape)
        
    def connect(self, layer):
        super(DenseLayer, self).connect(layer)
        self.weight = np.random.randn(self.dim, layer.dim)
        self.bias = np.random.randn(self.dim, 1)

        self.params = [self.weight, self.bias]

        self.delta_w = np.zeros(self.weight.shape)
        self.delta_b = np.zeros(self.bias.shape)


class Conv (Layer) :
    def __init__(self, weight , stride, pad , pool_filter_size, pool_stride, activations = None) :
        super(Conv, self).__init__(activations)  
        self.weight = weight
        self.stride = stride
        self.pool_filter_size = pool_filter_size
        self.pool_stride = pool_stride
        self.pad = pad
        
        self.bias = None
        self.params = [self.weight, self.bias]
        self.delta_w = None
        self.delta_b = None

    def im2col(self) :

        filter_count, channel, filter_height , filter_width = self.weight.shape
        input_count, channel, input_height, input_width = self.input.shape

        output_height = int((input_height +2*self.pad -filter_height )/self.stride)+1
        output_width= int ((input_width + 2*self.pad- filter_width)/self.stride)+1

        img = np.pad(self.input, [(0,0), (0,0), (self.pad, self.pad), (self.pad, self.pad)], 'constant')
        col = np.zeros((input_count, channel, filter_height, filter_width, output_height, output_width))

        for y in range(filter_height):
            y_max = y + self.stride*output_height
            for x in range(filter_width):
                x_max = x + self.stride*output_width
                col[:, :, y, x, :, :] = img[:, :, y:y_max:self.stride, x:x_max:self.stride]
            col = col.transpose(0, 4, 5, 1, 2, 3).reshape(input_count*output_height*output_width, -1)
        return col

    def MaxPooling(self) :
        raise NotImplementedError; 

    def feed_forward(self):
        self.input = self.get_input();
        input_count, channel, input_height, input_width = self.input.shape
        filter_count, channel, filter_height , filter_width = self.weight.shape
        output_height = int((input_height +2*self.pad -filter_height )/self.stride)+1
        output_width= int ((input_width + 2*self.pad- filter_width)/self.stride)+1
    
        # Convolution, Pooling 
        col=self.im2col(self.input)
        col_w=self.weight.reshape((filter_count, -1)).T

        temp=np.dot(col,col_w)+self.bias
        temp = self.activation_function.get_activate(temp);
        temp = self.MaxPooling()
        
        self.output = temp

    def backpropagation(self) :
        raise NotImplementedError; 

    def get_input() : 
        raise NotImplementedError

    def update(self, learning_rate):
        self.weight -= learning_rate * self.delta_w
        self.bias -= learning_rate * self.delta_b

    def clear_deltas(self):
        self.delta_w = np.zeros(self.weight.shape)
        self.delta_b = np.zeros(self.bias.shape)
        
    def connect(self, layer):
        raise NotImplementedError

# Conv -> Fully Connected Layer
class Flatten(Layer) :
    def __init__ (self) :
        raise NotImplementedError
