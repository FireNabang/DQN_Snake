import numpy as np


def mse(predictions, labels):
    diff = predictions - labels
    return 0.5 * sum(diff * diff)[0]

def mse_diff(predictions, labels):
    return predictions - labels    

loss_functions = {
  'mse': mse
}

loss_functions_diff = {
  'mse': mse_diff
}
    
    
    
class Loss:
    def __init__(self,_loss : str):
        self.loss = loss_functions[_loss]
        self.loss_diff = loss_functions_diff[_loss]

    def get_loss(self, predictions, labels):
        return self.loss(predictions, labels)

    
    def get_loss_diff(self, predictions, labels):
        return self.loss_diff(predictions,labels)