from six.moves import range
import random
import numpy as np
from nn.functions import loss




class SequentialNetwork:
    def __init__(self, _loss=None):
        if _loss is None:
            self.loss = loss.Loss('mse')
        else :
            self.loss = loss.Loss(_loss)

    
    def set_hidden_layer(self, layer):
        self.hidden_layer = layer
        
    def set_input_layer(self, layer):
        self.input_layer = layer
        
    def set_output_layer(self, layer):
        self.output_layer = layer
        
    
    def train(self, training_data, epochs, mini_batch_size,learning_rate, test_data=None):
        n = len(training_data)
        
        self.output_layer.connect(self.hidden_layer)
        self.hidden_layer.connect(self.input_layer)
        
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
        for _input,_output in mini_batch:
            self.feed_forward(_input)
            self.backpropagation(_output)
        self.update(mini_batch, learning_rate)

        
    def update(self, mini_batch, learning_rate):
        learning_rate = learning_rate / len(mini_batch)
        self.output_layer.update(learning_rate)
        self.hidden_layer.update(learning_rate)
        
        self.output_layer.clear_deltas()
        self.hidden_layer.clear_deltas()



    def backpropagation(self, _output):
        self.output_layer.output_delta = self.loss.get_loss_diff(self.output_layer.output, _output)
        self.output_layer.backpropagation()
        self.hidden_layer.backpropagation()
        
    def feed_forward(self,_input):
        self.input_layer.input  = _input
        self.input_layer.feed_forward()
        self.hidden_layer.feed_forward()
        self.output_layer.feed_forward()
        return self.output_layer.output

    def evaluate(self, test_data):
        test_results = [(
            np.argmax(self.feed_forward(x)),
            np.argmax(y)
        ) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
