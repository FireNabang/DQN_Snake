from collections import deque
import random
import numpy as np
from DNN.code.nn import network
from DNN.code.nn.layers import DenseLayer, Conv2DLayer, MaxPooling2DLayer, FlattenLayer,DropoutLayer


class DQNAgent:
    def __init__(self, field_size, batch_size, learning_rate, discount_factor):
        self.field_height, self.field_width = field_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.model_training = self.create_model()
        self.model = self.create_model()
        self.model.summary()

        self.game_data = deque(maxlen=65536)
        self.train_counter = 0

    def create_model(self):
        model = network.SequentialNetwork('categorical_crossentropy')
        model.add_layer(Conv2DLayer(input_size=(self.field_height,self.field_width,1)))
        model.add_layer(Conv2DLayer(filter_count=8,filter_size=(3,3)))
        model.add_layer(DropoutLayer(0.1))
        model.add_layer(Conv2DLayer(filter_count=8, filter_size=(3, 3)))
        model.add_layer(FlattenLayer())
        model.add_layer(DenseLayer(32, activations='sigmoid'))
        model.add_layer(DropoutLayer(0.1))
        model.add_layer(DenseLayer(4, activations='soft_max'))
        return model

    def update_game_data(self, current_state, action, reward, next_state, live):
        self.game_data.append((current_state, action, reward, next_state, live))

    def get_q_values(self, x):
        return self.model.predict(np.array([np.reshape(x, (1, self.field_height, self.field_width))]))

    def train(self):
        if len(self.game_data) < 512:
            return
        self.train_counter += 1
        samples = random.sample(self.game_data, 512)

        current_input = np.array([np.reshape(sample[0], (1, self.field_height,self.field_width)) for sample in samples])
        current_q_values = self.model.predict(current_input)

        next_input = np.array([np.reshape(sample[3], (1, self.field_height,self.field_width)) for sample in samples])
        next_q_values = self.model.predict(next_input)
        print(current_q_values)
        # update q values
        for i, (current_state, action, reward, _, live) in enumerate(samples):
            if live == 0:
                next_q_value = reward
            else:
                next_q_value = (1-self.learning_rate)*current_q_values[i][action] + self.learning_rate*(reward + self.discount_factor * np.max(next_q_values[i]))
            current_q_values[i][action] = next_q_value

        #  model train
        print("train_counter : " +str(self.train_counter))
        self.model_training.train(list(zip(current_input, current_q_values)), epochs=1, mini_batch_size=self.batch_size, learning_rate=0.9)

    def save(self, model_filepath):
        self.model.save_model(model_filepath)

    def load(self, model_filepath):
        self.model.load_model(model_filepath)