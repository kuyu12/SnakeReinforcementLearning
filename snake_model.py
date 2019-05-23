#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 16:29:50 2019

@author: yairhadad
"""
# keras 
from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier

from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
import random
import numpy as np
import pandas as pd
from operator import add


input_node = 9
        
class DQNAgent(object):

    def __init__(self):
        self.reward = 0
        self.gamma = 0.9
        self.dataframe = pd.DataFrame()
        self.short_memory = np.array([])
        self.agent_target = 1
        self.agent_predict = 0
        self.learning_rate = 0.0005
        self.model = self.network()
        #self.model = self.network("weights.hdf5")
        self.epsilon = 0

        self.actual = []
        self.memory = []

    def get_state(self, snake, apple):

        state = [
#            snake.checkCrashWithDirection(KEY["UP"]), # wall up
#            snake.checkCrashWithDirection(KEY["DOWN"]), # wall down
#            snake.checkCrashWithDirection(KEY["LEFT"]), # wall left
#            snake.checkCrashWithDirection(KEY["RIGHT"]), # wall right
            snake.direction == 1,  # move up
            snake.direction == 2,  # move down
            snake.direction == 3,  # move left
            snake.direction == 4,  # move right
            apple.x < snake.x,  # food left
            apple.x > snake.x,  # food right
            apple.y < snake.y,  # food up
            apple.y > snake.y,  # food down
            np.sqrt((apple.x-snake.x)**2 + (apple.y-snake.y)**2)
            ]

        for i in range(len(state)):
            if state[i]:
                state[i]=1
            else:
                state[i]=0

        return np.asarray(state)

    def set_reward(self, eaten, crash):
        self.reward = 0
        if crash:
            self.reward = -10
            return self.reward
        if eaten:
            self.reward = 10
        return self.reward

    def network(self, weights=None):
        model = Sequential()
        model.add(Dense(output_dim=120, activation='relu', input_dim=input_node))
        model.add(Dropout(0.15))
        model.add(Dense(output_dim=120, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(output_dim=120, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(output_dim=4, activation='softmax'))
        opt = Adam(self.learning_rate)
        model.compile(loss='mse', optimizer=opt)

        if weights:
            model.load_weights(weights)
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay_new(self, memory):
        if len(memory) > 1000:
            minibatch = random.sample(memory, 1000)
        else:
            minibatch = memory
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]))[0])
            target_f = self.model.predict(np.array([state]))
            target_f[0][np.argmax(action)] = target
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)

    def train_short_memory(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape((1, input_node)))[0])
        target_f = self.model.predict(state.reshape((1, input_node)))
        target_f[0][np.argmax(action)] = target
        self.model.fit(state.reshape((1, input_node)), target_f, epochs=1, verbose=0)




agent = DQNAgent()