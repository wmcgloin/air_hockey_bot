import sys
import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque

# main game files are in directory 'air_hockey' at the same level
game_dir = os.path.abspath('../air_hockey/')
sys.path.append(game_dir)

from air_hockey import AirHockeyGame  # Importing after adding the directory to path

class AirHockeyEnv(gym.Env):
    def __init__(self):
        super(AirHockeyEnv, self).__init__()
        self.game = AirHockeyGame()
        self.frame_buffer = deque(maxlen=3)  # Buffer to hold last three frames
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.game.screen_height, self.game.screen_width, 3 * 3), dtype=np.uint8)  # Stack 3 frames

    def reset(self):
        initial_state = self.game.reset()
        self.frame_buffer.clear()
        [self.frame_buffer.append(initial_state) for _ in range(3)]  # Fill buffer with initial state
        return np.concatenate(self.frame_buffer, axis=2)  # Concatenate frames along the channel dimension

    def step(self, action):
        reward, done, info = self.game.update_game(action)
        new_frame = self.game.get_state()
        self.frame_buffer.append(new_frame)
        obs = np.concatenate(self.frame_buffer, axis=2)
        return obs, reward, done, info

    def render(self, mode='human'):
        return self.game.draw()