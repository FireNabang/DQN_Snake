import random
import numpy as np
from nn.functions import loss


class SequentialNetwork:
    def __init__(self, _loss=None):
        self.input_layer = None
        if _loss is None:
            self.loss = loss.Loss('mse')
        else:
            self.loss = loss.Loss(_loss)

    def add_layer(self, layer):
        if self.input_layer is None:
            self.input_layer = layer
            self.output_layer = layer

        else:
            layer.connect(self.output_layer)
            self.output_layer = layer

    def train(self, training_data, epochs, mini_batch_size, learning_rate, test_data=None):
        n = len(training_data)
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.train_batch(mini_batch, learning_rate)

            if test_data:
                n_test = len(test_data)
                print("Epoch {0}: {1} / {2}"
                      .format(epoch, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(epoch))

    def train_batch(self, mini_batch, learning_rate):
        for _input, _output in mini_batch:
            self.feed_forward(_input)
            self.backpropagation(_output)
        self.update(mini_batch, learning_rate)

    def update(self, mini_batch, learning_rate):
        learning_rate = learning_rate / len(mini_batch)
        temp_layer = self.output_layer
        while True:
            temp_layer.update(learning_rate)
            temp_layer = temp_layer.previous_layer
            if temp_layer is self.input_layer:
                break
        temp_layer = self.output_layer
        while True:
            temp_layer.clear_deltas()
            temp_layer = temp_layer.previous_layer
            if temp_layer is self.input_layer:
                break

    def backpropagation(self, _output):
        self.output_layer.loss_delta = self.loss.get_loss_diff(self.output_layer.output, _output)
        temp_layer = self.output_layer
        while True:
            temp_layer.backpropagation()
            temp_layer = temp_layer.previous_layer
            if temp_layer is self.input_layer:
                break

    def feed_forward(self, _input):
        self.input_layer.input = _input
        temp_layer = self.input_layer
        while True:
            print('check')
            if temp_layer is None:
                break
            temp_layer.feed_forward()
            temp_layer = temp_layer.next_layer
        return self.output_layer.output

    def evaluate(self, test_data):
        test_results = [(
            np.argmax(self.feed_forward(x)),
            np.argmax(y)
        ) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
