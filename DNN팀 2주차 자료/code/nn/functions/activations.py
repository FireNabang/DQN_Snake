import numpy as np


def sigmoid(z):
    return np.reciprocal(np.add(1.0, np.exp(-z)))


def sigmoid_diff(z):
    return np.multiply(sigmoid(z), np.subtract(1, sigmoid(z)))

def none(z):
    return z

def ReLU(z):
    result = np.array([max(x,0) for x in z],dtype = np.float32)
    return r

def ReLU_diff(z):
    result = np.array([int(x>0) for x in z],dtype = np.float32)
    return result

def tanh(z):
    return 2*sigmoid(2*np.array(z)) - 1

def tanh_diff(z):
    return 1 - tanh(z)*tanh(z) 

def sofx_max(z):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def sofx_max_diff(z):
    result = np.diag(z)
    for i in range(len(result)):
        for j in range(len(result)):
            result[i][j] = s[i] * (int(i==j) - s[j])
            
    return result


activations = {
    'none' : none,
    'sigmoid': sigmoid,
    'tanh' : tanh,
    'relu' : ReLU,
    'softmax' : sofx_max
    
}

activations_diff = {
    'none' : none,
    'sigmoid': sigmoid_diff,
    'tanh' : tanh_diff,
    'relu' : ReLU_diff,
    'softmax' : sofx_max_diff
}

class ActivationFunction:
    def __init__(self, func:str):
        self.activation_function = activations[func]
        self.activation_function_diff = activations_diff[func]

    def get_activate(self,data):
        return self.activation_function(data)
        
    def get_activate_diff(self,data):
        return self.activation_function_diff(data)
