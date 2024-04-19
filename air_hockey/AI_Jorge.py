#AI_Jorge.py

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input

def create_model(input_shape, num_actions):
    model = models.Sequential([
        Input(shape=input_shape),  # Start model with an explicit Input layer
        layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu'),
        layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(num_actions)  # Output layer: one output for each possible action
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00025),
                  loss='mse')  # Mean Squared Error loss for regression
    return model


# DQN algorithm with experience replay
import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

def update_model(model, target_model, batch, gamma):
    states, actions, rewards, next_states, dones = zip(*batch)
    states = np.array(states)
    next_states = np.array(next_states)
    current_qs = model.predict(states)
    next_qs = target_model.predict(next_states)
    max_next_qs = np.max(next_qs, axis=1)
    targets = current_qs.copy()
    for idx, (reward, done, action) in enumerate(zip(rewards, dones, actions)):
        targets[idx, action] = reward + (gamma * max_next_qs[idx] * (1 - done))
    history = model.fit(states, targets, epochs=1, verbose=0)
    print(f"Training loss on latest update: {history.history['loss'][0]}")

def select_action(state, model, epsilon, num_actions):
    if random.random() < epsilon:
        # Randomly select an action for each paddle
        return (random.randint(0, num_actions - 1), random.randint(0, num_actions - 1))
    else:
        q_values = model.predict(state[np.newaxis, :])
        # Assume the model predicts actions for both paddles, split the output
        action1 = np.argmax(q_values[0][:num_actions//2])
        action2 = np.argmax(q_values[0][num_actions//2:])
        return (action1, action2)


def update_target_model(main_model, target_model):
    target_model.set_weights(main_model.get_weights())
