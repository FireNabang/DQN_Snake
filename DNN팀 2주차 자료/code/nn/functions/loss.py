import numpy as np


def mse(predictions, labels):
    diff = predictions - labels
    return 0.5 * sum(diff * diff)[0]

def mse_diff(predictions, labels):
    return predictions - labels    

def binary_crossentropy(predictions, labels):
    return -(labels * np.log(predictions) + (1-y)*np.log(1-predictions)).mean()

def binary_crossentropy_diff(predictions, labels):
    return -labels/predictions +(1-y)/(1-predictions)     

def categorical_crossentropy(predictions, labels):
    return -sum(labels * np.log(predictions))

def categorical_crossentropy_diff(predictions, labels):
    return -labels/predictions

loss_functions = {
    'mse': mse,
    'binary_crossentropy' : binary_crossentropy,
    'categorical_crossentropy' : categorical_crossentropy
}

loss_functions_diff = {
    'mse': mse_diff,
    'binary_crossentropy' : binary_crossentropy_diff,
    'categorical_crossentropy' : categorical_crossentropy_diff
}
    
    
    
class Loss:
    def __init__(self,_loss : str):
        self.loss = loss_functions[_loss]
        self.loss_diff = loss_functions_diff[_loss]

    def get_loss(self, predictions, labels):
        return self.loss(predictions, labels)

    
    def get_loss_diff(self, predictions, labels):
        return self.loss_diff(predictions,labels)