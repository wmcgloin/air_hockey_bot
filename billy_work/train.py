import os
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from air_hock_env import AirHockeyEnv

# Create the environment and wrap it for monitoring
env = Monitor(AirHockeyEnv())

# Optional: Check if the environment follows the Gym API
check_env(env)

# Using vectorized environments for better performance
vec_env = make_vec_env(lambda: env, n_envs=1)

# Create the DQN agent
model = DQN("CnnPolicy", vec_env, verbose=1, buffer_size=5000, learning_rate=1e-4, 
            exploration_fraction=0.1, exploration_final_eps=0.02, exploration_initial_eps=1.0)

# Train the agent
model.learn(total_timesteps=100000)

# Save the agent
model.save("dqn_air_hockey")

# Load the trained agent (optional, for demonstration here)
model = DQN.load("dqn_air_hockey", env=vec_env)

# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

# Enjoy trained agent
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()