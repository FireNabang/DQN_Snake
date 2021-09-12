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
            return np.array(self.input,dtype=np.float32)

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
