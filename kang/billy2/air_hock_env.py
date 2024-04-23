import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from collections import deque
from game import AirHockeyGame

class AirHockeyEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 3000}

    def __init__(self, render_mode = None):
        super(AirHockeyEnv, self).__init__()
        pygame.init()
        self.render_mode = render_mode
        self.screen = pygame.display.set_mode((800, 400))
        self.clock = pygame.time.Clock()
        self.game = AirHockeyGame(self.screen)
        # self.game.puck.launch_puck()
        self.frame_buffer = deque(maxlen=3)
        self.action_space = spaces.Discrete(9)  # Adjust as necessary to fit game control scheme
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.game.screen_width, self.game.screen_height, 3), dtype=np.uint8)
        self.verbose = False
        self.episode_counter = 0
        self.frame_save_interval = 1000

    def get_state(self):
        # Access the current game state from the AirHockeyGame instance
        # print(self.game.get_state())
        return self.game.get_state()

    # # FOR CNN IMPLEMENTATION
    # def step(self, action):
    #     reward, done, info = self.game.update(action)
    #     new_frame = self.game.get_state()
    #     self.update_frame_buffer(new_frame)
    #     obs = np.concatenate(list(self.frame_buffer), axis=2)
    #     if self.episode_counter % self.frame_save_interval == 0:
    #         np.save(f'frame_data_episode_{self.episode_counter}.npy', obs)
    #     self.render()
    #     return obs, reward, done, {'truncated': False}, info

    # FOR Linear IMPLEMENTATION
    def step(self, action):
        reward, done, info = self.game.update(action)
        obs = self.game.get_state()
        if self.episode_counter % self.frame_save_interval == 0:
            np.save(f'frame_data_episode_{self.episode_counter}.npy', obs)

        self.render()
        return obs, reward, done, {'truncated': False}, info


    # # FOR CNN IMPLEMENTATION
    # def reset(self, seed=None, options=None):
    #     super().reset(seed=seed)
    #     initial_state = self.game.reset()
    #     self.frame_buffer.clear()
    #     for _ in range(3):
    #         self.frame_buffer.append(initial_state)
    #     observation = np.concatenate(list(self.frame_buffer), axis=2)
    #     self.episode_counter += 1
    #     return observation, {}
    
    # FOR Linear IMPLEMENTATION
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        observation = self.game.reset()
        return observation, {}

    # def update_frame_buffer(self, frame):
    #     if len(self.frame_buffer) < 3:
    #         while len(self.frame_buffer) < 3:
    #             self.frame_buffer.append(frame)
    #     else:
    #         self.frame_buffer.append(frame)

    def render(self):
        if self.render_mode == 'human':
            self.game.draw()  # Assume this method draws the current game state
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata['render_fps'])
        else:
            pass

    def close(self):
        pygame.quit()


