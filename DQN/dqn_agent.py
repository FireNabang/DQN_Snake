from collections import deque
import random
import numpy as np
from DNN.code.nn import network
from DNN.code.nn.layers import DenseLayer, Conv2DLayer, MaxPooling2DLayer, FlattenLayer,DropoutLayer


class DQNAgent:
    def __init__(self, field_size, batch_size, learning_rate, discount_factor, epochs, data_min_size):
        self.field_height, self.field_width = field_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epochs = epochs
        self.data_min_size = data_min_size

        self.model_training = self.create_model()
        self.model = self.create_model()
        self.model.summary()

        self.game_data = deque(maxlen=(data_min_size*3 //2))
        self.train_counter = 0

    def create_model(self):
        model = network.SequentialNetwork()
        model.add_layer(Conv2DLayer(input_size=(self.field_height,self.field_width,1)))
        model.add_layer(Conv2DLayer(filter_count=16,filter_size=(2,2)))
        model.add_layer(FlattenLayer())
        model.add_layer(DenseLayer(128, activations='sigmoid'))
        model.add_layer(DenseLayer(4))
        return model

    def update_game_data(self, current_state, action, reward, next_state, live):
        self.game_data.append((current_state, action, reward, next_state, live))

    def get_q_values(self, x):
        return self.model.predict(np.array([np.reshape(x, (1, self.field_height, self.field_width))]) )

    def train(self):
        if len(self.game_data) < self.data_min_size:
            return
        self.train_counter += 1
        samples = random.sample(self.game_data, self.data_min_size)

        current_input = np.array([np.reshape(sample[0], (1, self.field_height,self.field_width)) for sample in samples])
        current_q_values = self.model.predict(current_input)

        next_input = np.array([np.reshape(sample[3], (1, self.field_height,self.field_width)) for sample in samples])
        next_q_values = self.model.predict(next_input)
        # update q values
        for i, (current_state, action, reward, _, live) in enumerate(samples):
            if live == 0:
                next_q_value = reward
            else:
                next_q_value = current_q_values[i][action] + self.learning_rate*(reward + self.discount_factor * np.max(next_q_values[i]))
            current_q_values[i][action] = next_q_value

        #  model train
        print("train_count : " + str(self.train_counter))
        self.model_training.train(list(zip(current_input, current_q_values)), epochs=self.epochs, mini_batch_size=self.batch_size, learning_rate=0.9)

    def save(self, model_filepath):
        self.model.save_model(model_filepath)

    def load(self, model_filepath):
        self.model.load_model(model_filepath)