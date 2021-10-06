import random
import numpy as np
import h5py
from nn.functions import loss


class SequentialNetwork:
    def __init__(self, _loss=None):
        self.model_loaded = False
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
            temp_layer.update(learning_rate, len(mini_batch))
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
            if temp_layer is None:
                break
            temp_layer.feed_forward()
            temp_layer = temp_layer.next_layer
        return self.output_layer.output

    def summary(self):
        temp_layer = self.input_layer
        print("{0:<20} | {1:<20}".format('Layer', 'Output Shape'))
        print("-------------------------------------------")
        while True:
            if temp_layer is None:
                break
            print('{0:<20} | {1:<20}'.format(temp_layer.__class__.__name__, str(temp_layer.get_shape())))
            temp_layer = temp_layer.next_layer

    def save_model(self, file='model'):
        f = h5py.File(file + '.hdf5', 'w')
        idx = 1
        temp_layer = self.input_layer.next_layer
        while True:
            idx += 1
            if temp_layer is None:
                break
            if temp_layer.params is None:
                temp_layer = temp_layer.next_layer
                continue
            f.create_dataset(name='Layer'+str(idx), shape = temp_layer.params[0].shape, dtype = np.float32, data = temp_layer.params[0].astype(np.float32))
            temp_layer = temp_layer.next_layer
        f.close()

    def load_model(self, file='model'):
        try:
            f = h5py.File(file + '.hdf5', 'r')
        except:
            return
        idx = 0
        temp_layer = self.input_layer
        while True:
            idx += 1
            if temp_layer is None:
                break
            if 'Layer'+str(idx) in f.keys() :
                temp_layer.params[0] = np.array(f.get('Layer'+str(idx))[:] , dtype = np.float32)
            temp_layer = temp_layer.next_layer
        f.close()
        self.model_loaded = True
        print('model is loaded')
        

    def evaluate(self, test_data):
        test_results = [(
            np.argmax(self.feed_forward(x)),
            np.argmax(y)
        ) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
