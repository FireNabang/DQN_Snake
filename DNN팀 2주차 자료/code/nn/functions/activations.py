import numpy as np


def sigmoid(z):
    return np.reciprocal(np.add(1.0, np.exp(-z)))


def sigmoid_diff(z):
    return np.multiply(sigmoid(z), np.subtract(1, sigmoid(z)))

def none(z):
    return z


activations = {
    'none' : none,
    'sigmoid': sigmoid
    
}

activations_diff = {
    'none' : none,
    'sigmoid': sigmoid_diff
}

class ActivationFunction:
    def __init__(self, func:str):
        self.activation_function = activations[func]
        self.activation_function_diff = activations_diff[func]

    def get_activate(self,data):
        return self.activation_function(data)
        
    def get_activate_diff(self,data):
        return self.activation_function_diff(data)
