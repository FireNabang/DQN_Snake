import numpy as np

delta = 1e-4


def mse(predictions, labels):
    diff = predictions - labels
    return 0.5 * sum(diff * diff)[0]


def mse_diff(predictions, labels):
    return predictions - labels


def binary_crossentropy(predictions, labels):
    return -(labels * np.log(predictions + delta) + (1-labels)*np.log(1-predictions + delta)).mean()


def binary_crossentropy_diff(predictions, labels):
    return -labels/(predictions + delta) + (1 - labels) / (1 - predictions + delta)


def categorical_crossentropy(predictions, labels):
    return -sum(labels * np.log(predictions + delta))


def categorical_crossentropy_diff(predictions, labels):
    return - (labels / (predictions + delta))


loss_functions = {
    'mse': mse,
    'binary_crossentropy': binary_crossentropy,
    'categorical_crossentropy': categorical_crossentropy
}

loss_functions_diff = {
    'mse': mse_diff,
    'binary_crossentropy': binary_crossentropy_diff,
    'categorical_crossentropy': categorical_crossentropy_diff
}


class Loss:
    def __init__(self, _loss: str):
        self.loss = loss_functions[_loss]
        self.loss_diff = loss_functions_diff[_loss]

    def get_loss(self, predictions, labels):
        return self.loss(predictions, labels)

    def get_loss_diff(self, predictions, labels):
        return self.loss_diff(predictions, labels)
