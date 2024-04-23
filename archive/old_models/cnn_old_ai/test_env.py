import warnings
warnings.filterwarnings("ignore")

from gymnasium.utils.env_checker import check_env
from air_hock_env import AirHockeyEnv

env = AirHockeyEnv()
check_env(env)