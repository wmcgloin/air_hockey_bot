import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from collections import deque
from game import AirHockeyGame

class AirHockeyEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, render_mode = None):
        super(AirHockeyEnv, self).__init__()
        pygame.init()
        self.render_mode = render_mode
        self.screen = pygame.display.set_mode((800, 400))
        self.clock = pygame.time.Clock()
        self.game = AirHockeyGame(self.screen)
        self.frame_buffer = deque(maxlen=3)
        self.action_space = spaces.Discrete(9)  # Adjust as necessary to fit game control scheme
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.game.screen_width, self.game.screen_height, 3), dtype=np.uint8)
        self.verbose = False
        self.episode_counter = 0
        self.frame_save_interval = 1000

    def get_state(self):
        # Access the current game state from the AirHockeyGame instance
        return self.game.get_state()

    def step(self, action):
        reward, done, info = self.game.update(action)
        new_frame = self.game.get_state()
        self.update_frame_buffer(new_frame)
        obs = np.concatenate(list(self.frame_buffer), axis=2)
        if self.episode_counter % self.frame_save_interval == 0:
            np.save(f'frame_data_episode_{self.episode_counter}.npy', obs)
        self.render()
        return obs, reward, done, {'truncated': False}, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        initial_state = self.game.reset()
        self.frame_buffer.clear()
        for _ in range(3):
            self.frame_buffer.append(initial_state)
        observation = np.concatenate(list(self.frame_buffer), axis=2)
        self.episode_counter += 1
        return observation, {}

    def update_frame_buffer(self, frame):
        if len(self.frame_buffer) < 3:
            while len(self.frame_buffer) < 3:
                self.frame_buffer.append(frame)
        else:
            self.frame_buffer.append(frame)

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




# import gymnasium as gym
# from gymnasium import spaces
# import numpy as np
# from collections import deque
# from game import AirHockeyGame
# import pygame

# class AirHockeyEnv(gym.Env):
#     def __init__(self):
#         super(AirHockeyEnv, self).__init__()
#         pygame.init()  # Initialize Pygame
#         screen = pygame.display.set_mode((800, 400))  # Create a Pygame screen
#         self.game = AirHockeyGame(screen)  # Pass the screen to AirHockeyGame
#         self.frame_buffer = deque(maxlen=3)  # Buffer to hold last three frames
#         self.action_space = spaces.Discrete(9)
#         self.observation_space = spaces.Box(low=0, high=255, shape=(self.game.screen_width, self.game.screen_height, 3), dtype=np.uint8)  # Stack 3 frames
#         self.verbose = False

#     def get_state(self):
#         # Access the current game state from the AirHockeyGame instance
#         return self.game.get_state()

#     def step(self, action, truncated = True):
#         reward, done, info = self.game.update(action) # Update the game with the current action
#         new_frame = self.game.get_state() # Get the new frame after the action
#         if len(self.frame_buffer) < 3:
#             # If we don't have enough frames yet, repeat the new frame
#             while len(self.frame_buffer) < 3:
#                 self.frame_buffer.append(new_frame)
#         else:
#             # Otherwise, add the new frame to the buffer and pop the oldest
#             self.frame_buffer.append(new_frame)
        
#         # Now we concatenate along the channel dimension to form the observation
#         obs = np.concatenate(list(self.frame_buffer), axis=2)
#         assert(obs.shape == self.observation_space.shape), str(obs.shape)
#         self.render()
#         return obs, reward, done, truncated, info
        
#     def reset(self, seed=None, options=None):
#         super().reset(seed=seed)
#         initial_state = self.game.reset()
#         self.frame_buffer.clear()
#         [self.frame_buffer.append(initial_state) for _ in range(3)]  # Fill buffer with initial state
#         observation = np.concatenate(self.frame_buffer, axis=2)  # Concatenate frames along the channel dimension
#         if self.verbose:
#             print("Observation shape: ", observation.shape)
#             print(observation)
#             print(self.observation_space)
#         info = {}  # Optionally, you can include additional reset information here
#         assert(observation.shape == self.observation_space.shape), str(observation.shape)
#         return observation, info  # Return a tuple of observation and info

#     def render(self, mode='human'):
#         return self.game.draw()
#     # i want to save the arrays of every frame for every 1000 episodes, so that i can visualize the learning process
    
#     def close(self): # Clean up resources
#         self.game.quit()