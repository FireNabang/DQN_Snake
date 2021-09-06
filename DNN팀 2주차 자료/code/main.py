from nn import load_mnist
from nn import network
from nn.layers import DenseLayer

training_data, test_data = load_mnist.load_data()

model = network.SequentialNetwork()

model.set_input_layer(DenseLayer(784, 1 , 'none'))
model.set_hidden_layer(DenseLayer(784, 196,'sigmoid'))
model.set_output_layer(DenseLayer(196, 10,'sigmoid'))


model.train(training_data, epochs=3, mini_batch_size=10,
          learning_rate=3.0, test_data=test_data)
