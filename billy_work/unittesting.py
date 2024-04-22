import unittest
import numpy as np
from air_hock_env import AirHockeyEnv  # Assuming the class is in this module

class TestAirHockeyEnv(unittest.TestCase):

    def test_initialization(self):
        env = AirHockeyEnv()
        self.assertIsInstance(env, AirHockeyEnv)
        self.assertEqual(len(env.frame_buffer), 3)

    def test_step_function(self):
        env = AirHockeyEnv()
        initial_observation, _ = env.reset()
        self.assertEqual(initial_observation.shape, (env.game.screen_width, env.game.screen_height, 9))

        action = env.action_space.sample()  # Sample a random action
        observation, reward, done, _, info = env.step(action)
        self.assertEqual(observation.shape, (env.game.screen_width, env.game.screen_height, 9))
        self.assertIn('truncated', info)

    def test_reset_function(self):
        env = AirHockeyEnv()
        observation, _ = env.reset()
        self.assertEqual(len(env.frame_buffer), 3)
        self.assertEqual(observation.shape, (env.game.screen_width, env.game.screen_height, 9))

    def test_render_mode(self):
        env = AirHockeyEnv(render_mode='human')
        env.render()  # Should not raise an error

    def test_frame_buffer_update(self):
        env = AirHockeyEnv()
        env.reset()
        new_frame = np.random.randint(0, 255, (env.game.screen_width, env.game.screen_height, 3), dtype=np.uint8)
        env.update_frame_buffer(new_frame)
        self.assertEqual(len(env.frame_buffer), 3)

if __name__ == '__main__':
    unittest.main()
