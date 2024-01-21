import math
import os.path

import numpy as np
from keras import Sequential, Input
from keras.src.layers import Dense, Flatten, Reshape, Conv2D, MaxPooling2D, Activation, Dropout
from keras.src.optimizers import Adam
from keras import models
import random
from collections import deque
from mtensorboard import ModifiedTensorBoard
import time


class DQNAgent:
    def __init__(self, state_size, action_size, name):
        self.state_size = state_size
        self.action_tuple_size = action_size
        self.action_size = action_size[0]*action_size[1]
        self.memory = deque(maxlen=50000)
        self.tensorboard = ModifiedTensorBoard(
            log_dir=f"logs/{name}-{int(time.time())}")
        self.target_update_counter = 0
        self.gamma = 0.99  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        if os.path.isfile('models/'+name+".keras"):
            print("Loading saved models for agent:", name)
            self.model = models.load_model('./models/'+name+".keras")
            self.target_model = models.load_model(
                './models/'+name+"_target.keras")
        else:
            self.model = self._build_model()
            self.target_model = self._build_model()
            self.update_target_model()
        # main model is self.model, this gets trained each step of the env.
        # target model is what we .predict against each step of the env.
        self.name = name

    def _build_model(self):
        model = Sequential()
        model.add(Conv2D(128, (3, 3), input_shape=self.state_size))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2,2))
        model.add(Dropout(0.2))

        model.add(Conv2D(128, (3, 3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2,2))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Dense(32))
        model.add(Dense(self.action_size, activation='linear'))
        model.add(Reshape(self.action_tuple_size))
        model.compile(loss='mse', optimizer=Adam(
            learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        self.target_update_counter = 0

    def update_replay_memory(self, transition):
        self.memory.append(transition)

    def get_qs(self, state, step):
        return self.model_predict(np.array(state).reshape(-1, *state.shape))[0]

    def train(self, terminal_state, step):
        if len(self.memory) < 10000:
            return
        minibatch = random.sample(self.memory, 16)
        # Get current states before action is applied.
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)
        for x in range(len(current_states)):
            for i in range(len(current_qs_list[x])):
                for j in range(len(current_qs_list[x][i])):
                    if current_states[x][i][j] != -1:
                        # if already clicked then make a really low reward as to prevent clicking on this.
                        current_qs_list[x][i][j] = -math.inf
        # get after action is applied.
        new_current_states = np.array(
            [transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)
        for x in range(len(new_current_states)):
            for i in range(len(future_qs_list[x])):
                for j in range(len(future_qs_list[x][i])):
                    if new_current_states[x][i][j] != -1:
                        # if already clicked then make a really low reward as to prevent clicking on this.
                        future_qs_list[x][i][j] = -math.inf
        X = []
        y = []
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.gamma*max_future_q
            else:
                new_q = reward
            current_qs = current_qs_list[index]
            current_qs[action] = new_q
            X.append(current_state)
            y.append(current_qs)
        self.model.fit(np.array(X), np.array(y), batch_size=16, verbose=0,
                       shuffle=False if terminal_state else None)
        if terminal_state:
            self.target_update_counter += 1
            if self.target_update_counter > 5:
                self.update_target_model()

    def act(self, state, env):
        if np.random.rand() <= self.epsilon:
            available_pos = []
            for i in range(len(env.grid)):
                for j in range(len(env.grid[i])):
                    if env.grid[i][j] == -1:
                        available_pos.append((i, j))
            return available_pos[random.randint(0, len(available_pos)-1)]
        state = np.expand_dims(state, axis=0)
        act_values = self.model.predict(state, verbose=0)[0]
        for i in range(len(act_values)):
            for j in range(len(act_values[i])):
                if env.grid[i][j] != -1:
                    # if already clicked then make a really low reward as to prevent clicking on this.
                    act_values[i][j] = -math.inf
        x = np.unravel_index(np.argmax(act_values), act_values.shape)
        return x

    def act_target(self, state, env):
        if np.random.rand() <= self.epsilon:
            available_pos = []
            for i in range(len(env.grid)):
                for j in range(len(env.grid[i])):
                    if env.grid[i][j] == -1:
                        available_pos.append((i, j))
            return np.random.choice(available_pos)
        state = np.expand_dims(state, axis=0)
        act_values = self.target_model.predict(state, verbose=0)[0]
        for i in range(len(act_values)):
            for j in range(len(act_values[i])):
                if env.grid[i][j] != -1:
                    # if already clicked then make a really low reward as to prevent clicking on this.
                    act_values[i][j] = -math.inf
        x = np.unravel_index(np.argmax(act_values), act_values.shape)
        return x

    def save(self):
        self.model.save('./models/'+self.name+".keras")
        self.target_model.save('./models/' + self.name + "_target.keras")
