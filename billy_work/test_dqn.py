# test_dqn.py
import warnings
warnings.filterwarnings("ignore")

from air_hock_env import AirHockeyEnv
from dqn import DQNAgent

env = AirHockeyEnv()
agent = DQNAgent(env)
agent.train(num_episodes=100)