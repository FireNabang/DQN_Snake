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

        
    def get_input(self):
        raise NotImplementedError
        
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

    def get_input(self):
        if self.previous_layer is not None:
            return np.dot(self.weight, self.previous_layer.output) + self.bias
        else:
            return np.array(self.input,dtype=np.float128)
        
    def feed_forward(self):
        self.input = self.get_input()
        self.output = self.activation_function.get_activate(self.input)
        
    def backpropagation(self):
        data = self.get_input()
        accumulated_delta = self.get_accumulated_delta()
        activate_diff = self.get_activate_diff(data)
        # print(accumulated_delta)

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


class Conv2DLayer (Layer) :
    def __init__(self,filter_count, filter_size, pad, activations = None,stride=1,input_size = (1,1)) :
        super(Conv2DLayer, self).__init__(activations)  
        self.stride = stride
        
        self.filter_height = filter_size[0]
        self.filter_width = filter_size[1]
        self.filter_count = filter_count
        
        self.pad = pad
        self.input_height = input_size[0]
        self.input_width = input_size[1]
        self.output_height = None 
        self.output_width = None
        
        self.output_channel = None
        self.input_channel = None
        
        self.weight = np.array([np.random.randn(self.filter_height,self.filter_width) for _ in range(self.filter_count)], dtype  = np.float128)
        self.bias = np.array([np.random.randn(1,1) for _ in range(self.filter_count)], dtype = np.float128) 
        self.params = [self.weight, self.bias]
        self.delta_w = None
        self.delta_b = None
        
        
    def get_input(self):
        img = np.pad(self.previous_layer.output, [(0,0), (0,0), (self.pad, self.pad), (self.pad, self.pad)], 'constant')
        if self.previous_layer is not None:
            result = np.array([np.zeros((self.input_height,self.output_width)) for _ in range(self.input_channel)], dtype = np.float128)
            for pre_output_ch in range(self.previous_layer.output_channel):
                for filter_cnt in range(self.filter_count):
                    for m in range(0,self.input_width - self.filter_width,self.stride):
                        for n in range(0,self.input_height - self.filter_height,self.stride):
                            self.input[pre_output_ch * self.filter_count + filter_cnt][n:n+self.filter_height][m:m+self.filter_width] += self.weight[filter_cnt] * img[n:n+self.filter_height][m:m+self.filter_width] + self.bias[filter_cnt]

            return result
        else:
            return np.array(self.input,dtype=np.float128)
        

    def feed_forward(self):
        self.input_height = (self.previous_layer.output_height  + 2 * self.pad - self.filter_height ) // self.stride + 1
        self.input_width =  (self.previous_layer.output_width + 2 * self.pad - self.filter_width) // self.stride + 1
        self.input = self.get_input();
        self.ouptut = self.activation_function.get_activate(self.input)

    def backpropagation(self) :
        raise NotImplementedError


    def update(self, learning_rate):
        self.weight -= learning_rate * self.delta_w
        self.bias -= learning_rate * self.delta_b

    def clear_deltas(self):
                              
        self.delta_w = np.zeros(self.weight.shape)
        self.delta_b = np.zeros(self.bias.shape)
        
                              
    def connect(self, layer):
        super(Conv2DLayer, self).connect(layer)
        self.input_channel = layer.output_channel
        self.output_channel = self.input_channel * self.filter_count


# Conv -> Fully Connected Layer

class MaxPooling2DLayer(Layer):
    def __init__(self,pool_size=(2,2),activations = None):

        super(MaxPooling2DLayer, self).__init__(activations)
        self.pool_size = pool_size
        self.pool_height = pool_size[0]
        self.pool_width = pool_size[1]
        
        
        self.input_height = None
        self.input_width = None
        self.output_height = None 
        self.output_width = None
        
        self.channel = None
        
        
    def get_input(self):
        if self.previous_layer is not None:
            temp = self.previous_layer.output
            result = np.zeros(self.input_channel, self.input_height, self.input_width)
            for pre_output_ch in range(self.previous_layer.output_channel):
                for m in range(self.input_width):
                    for n in range(self.input_height):
                        result[pre_output_ch][n][m] += max([temp[pre_output_ch][n + i][m + j] for i in range(self.pool_height) for j in range(self.pool_width)])
            return result
        else:
            return None

    def feed_forward(self):
        self.input_height = (self.previous_layer.output_height  - self.pool_height) + 1
        self.input_width =  (self.previous_layer.output_width  - self.pool_widht) + 1
        self.channel = self.previous_layer.output_channel
        self.input = self.get_input()
        self.output = self.input
        
    def backpropagation(self):
        accumulated_delta = self.get_accumulated_delta()
        delta = np.zeros(self.previous_layer.output.shape)
        
        temp = self.previous_layer.output
        for ch in range(self.channel):
            for n in range(self.input_height):
                for m in range(self.input_width):
                    delta[ch][n][m] += accumulated_delta[ch][n][m] * int(self.input[ch][n][m] == max([temp[ch][n + i][m + j] for i in range(self.pool_height) for j in range(self.pool_width)]))
                    
        
        self.accumulated_delta = delta


    def update(self, learning_rate):
        pass

    def clear_deltas(self):
        pass
        
    def connect(self, layer):
        super(DenseLayer, self).connect(layer)


class AvgPooling2DLayer(Layer):
    def __init__(self,pool_size=(2,2),activations = None):

        super(AvgPooling2DLayer, self).__init__(activations)
        self.pool_size = pool_size
        self.pool_height = pool_size[0]
        self.pool_width = pool_size[1]
        
        
        self.input_height = None
        self.input_width = None
        self.output_height = None 
        self.output_width = None
        
        self.channel = None
        
        
    def get_input(self):
        if self.previous_layer is not None:
            temp = self.previous_layer.output
            result = np.zeros(self.input_channel, self.input_height, self.input_width)
            for pre_output_ch in range(self.previous_layer.output_channel):
                for m in range(self.input_width):
                    for n in range(self.input_height):
                        result[pre_output_ch][n][m] += sum([temp[ch][n + i][m + j] for i in range(self.pool_height) for j in range(self.pool_width)]) / (self.pool_height * self.pool_width)
            return result
        else:
            return None

    def feed_forward(self):
        self.input_height = (self.previous_layer.output_height  - self.pool_height) + 1
        self.input_width =  (self.previous_layer.output_width  - self.pool_widht) + 1
        self.channel = self.previous_layer.output_channel
        self.input = self.get_input()
        self.output = self.input
        
    def backpropagation(self):
        accumulated_delta = self.get_accumulated_delta()
        delta = np.zeros(self.previous_layer.output.shape)
        
        temp = self.previous_layer.output
        for ch in range(self.channel):
            for n in range(self.input_height):
                for m in range(self.input_width):
                    delta[ch][n][m] += accumulated_delta[ch][n][m] * sum([temp[ch][n + i][m + j] for i in range(self.pool_height) for j in range(self.pool_width)]) / (self.pool_height * self.pool_width)
                    
        
        self.accumulated_delta = delta


    def update(self, learning_rate):
        pass

    def clear_deltas(self):
        pass
        
    def connect(self, layer):
        super(DenseLayer, self).connect(layer)
    

class FlattenLayer(Layer):
    def __init__(self,dim=None,activations = None):

        super(DenseLayer, self).__init__(activations)
        self.dim = dim
        
        self.params = [self.weight, self.bias]

        self.delta_w = None
        self.delta_b = None
        
    def get_input(self):
        if self.previous_layer is not None:
            return self.previous_layer.output
        else:
            return None
    def feed_forward(self):
        self.input = self.get_input()
        self.output = self.input.reshape(self.previous_layer.output_width * self.previous_layer.output_height * self.previous_layer.output_channel,-1)
        
    def backpropagation(self):
        data = self.get_input()
        accumulated_delta = self.get_accumulated_delta()
        activate_diff = self.get_activate_diff(data)

        if self.next_layer is None and (accumulated_delta.shape != activate_diff.shape) :
            delta = np.dot(activate_diff.transpose(), accumulated_delta)
        else:
            delta = accumulated_delta * activate_diff
        
        self.accumulated_delta = np.dot(self.weight.transpose(), delta)

    '''
    def im2col(self) :

        filter_count, channel, filter_height , filter_width = self.weight.shape
        input_channel, channel, input_height, input_width = self.input.shape

        output_height = int((input_height +2*self.pad -filter_height )/self.stride)+1
        output_width= int ((input_width + 2*self.pad- filter_width)/self.stride)+1

        img = np.pad(self.input, [(0,0), (0,0), (self.pad, self.pad), (self.pad, self.pad)], 'constant')
        col = np.zeros((input_channel, channel, filter_height, filter_width, output_height, output_width))

        for y in range(filter_height):
            y_max = y + self.stride*output_height
            for x in range(filter_width):
                x_max = x + self.stride*output_width
                col[:, :, y, x, :, :] = img[:, :, y:y_max:self.stride, x:x_max:self.stride]
            col = col.transpose(0, 4, 5, 1, 2, 3).reshape(input_channel*output_height*output_width, -1)
        return col
        '''

    def update(self, learning_rate):
        pass

    def clear_deltas(self):
        pass
        
    def connect(self, layer):
        super(DenseLayer, self).connect(layer)
