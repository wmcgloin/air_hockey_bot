# air_hock_env.py
import sys
import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
# import gym 
# from gym import spaces
from collections import deque
from game import AirHockeyGame
import pygame

class AirHockeyEnv(gym.Env):
    def __init__(self):
        super(AirHockeyEnv, self).__init__()
        pygame.init()  # Initialize Pygame
        screen = pygame.display.set_mode((800, 400))  # Create a Pygame screen
        self.game = AirHockeyGame(screen)  # Pass the screen to AirHockeyGame
        self.frame_buffer = deque(maxlen=3)  # Buffer to hold last three frames
        self.action_space = spaces.Discrete(9)
        # If stacking along time axis but still treating each frame as independent in sequence
        self.observation_space = spaces.Box(low=0, high=255, shape=(800, 400, 3), dtype=np.uint8)

        self.verbose = False

    def get_state(self):
        # Access the current game state from the AirHockeyGame instance
        return self.game.get_state()

    def step(self, action):
        reward, done, info = self.game.update(action)  # Update the game with the current action
        new_frame = self.game.get_state()  # Get the new frame after the action
        
        if len(self.frame_buffer) < 3:
            # If we don't have enough frames yet, repeat the new frame
            while len(self.frame_buffer) < 3:
                self.frame_buffer.append(new_frame)
        else:
            # Otherwise, add the new frame to the buffer and pop the oldest
            self.frame_buffer.append(new_frame)

        # Adjust here to use the latest frame directly or process buffer differently
        obs = np.array(self.frame_buffer)[-1]  # Directly use the latest frame as the observation
        return obs, reward, done, {}, info
        
    def reset(self):
        super().reset()
        initial_state = self.game.reset()
        self.frame_buffer.clear()
        [self.frame_buffer.append(initial_state) for _ in range(3)]  # Fill buffer with initial state
        return self.frame_buffer[-1]

    def render(self, mode='human'):
        return self.game.draw()
    
    def close(self): # Clean up resources
        self.game.quit()