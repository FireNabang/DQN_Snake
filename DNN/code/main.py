from nn import load_mnist
from nn import network
from nn.layers import DenseLayer, Conv2DLayer, MaxPooling2DLayer, FlattenLayer,DropoutLayer

def main():
    training_data, test_data = load_mnist.load_data(train_data_cnt=1000, test_data_cnt=1000, shape='matrix')
    model = network.SequentialNetwork('categorical_crossentropy')
    model.add_layer(DenseLayer(784))
    model.add_layer(DenseLayer(256, activations='sigmoid'))
    model.add_layer(DenseLayer(64, activations='sigmoid'))
    model.add_layer(DenseLayer(10, activations='soft_max'))
    
    # model.load_model()

    model.summary()
    model.train(training_data, epochs=3, mini_batch_size=10, learning_rate=0.9, test_data=test_data)

    model.save_model('test1')


if __name__ == '__main__':
    main()