import os
import random
import time
from dataclasses import dataclass

# import gymnasium as gym
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pong_v2_env import PongV2

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

@dataclass
class Config:
    seed: int = 1
    episodes: int = 10000000
    learning_rate: float = 0.0005
    gamma: float = 0.99  # Discount factor for future rewards
    epsilon_start: float = 1.0  # Start value of epsilon
    epsilon_end: float = 0.01  # Minimum value of epsilon
    epsilon_decay: float = 30000  # Decay rate of epsilon
    buffer_size: int = 50000  # Capacity of the replay buffer
    batch_size: int = 32  # Size of batch taken from replay buffer
    target_update: int = 1000  # Frequency of target net update
    # env_id: str = 'AirHockey-v0'
    video_interval: int = 10000
    video_path: str = './videos/'

def make_env(cfg):
    env = PongV2()
    # env.seed(cfg.seed)
    return env

def capture_frame(frame, video_writer):
    video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

def train(cfg):
    verbose = False
    env = make_env(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DQN(env.observation_space.shape[0], env.action_space.n).to(device)
    target_model = DQN(env.observation_space.shape[0], env.action_space.n).to(device)
    target_model.load_state_dict(model.state_dict())
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    replay_buffer = []
    state = env.reset()
    if verbose:
        print(state)
        print("LenState: ", len(state))
        print("Observation Space: ", env.observation_space.shape[0])
    state = np.reshape(state, [1, env.observation_space.shape[0]])
    
    # Video recording
    # video_writer = None
    # frame_number = 0
    # episode_number = 0
    
    steps_done = 0
    i1 =0
    for episode in range(cfg.episodes):
        # if episode_number % cfg.video_interval == 0:
        #     if video_writer:
        #         video_writer.release()

        #     video_filename = os.path.join(cfg.video_path, f'episode_{episode_number}.avi')
        #     fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #     video_writer = cv2.VideoWriter(video_filename, fourcc, 30, (640, 480))
        if i1 % 1000 == 0:
            env.render(mode='human')
        epsilon = cfg.epsilon_end + (cfg.epsilon_start - cfg.epsilon_end) * \
                  np.exp(-1. * steps_done / cfg.epsilon_decay)
        if random.random() > epsilon:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(device)
                action = model(state_tensor).argmax().item()
        else:
            action = env.action_space.sample()

        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, env.observation_space.shape[0]])
        replay_buffer.append((state, action, reward, next_state, done))
        if not done:
            state = next_state
        else:
            state = env.reset()
            state = np.reshape(state, [1, env.observation_space.shape[0]])
        # state = next_state if not done else env.reset()
        # state = np.reshape(state, [1, env.observation_space.shape[0]])

        if len(replay_buffer) > cfg.buffer_size:
            replay_buffer.pop(0)

        if steps_done % cfg.target_update == 0:
            target_model.load_state_dict(model.state_dict())

        # Sample a batch for training
        if len(replay_buffer) > cfg.batch_size:
            # for item in replay_buffer:
            #     print("item:", len(item[0]))
            batch = random.sample(replay_buffer, cfg.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            
            # DEBUG PRINTS

            # print("\n\n")
            # print(states)
            # write a small loop that verifys all the states are the same size
            # print("\n\n")

            # c = 0
            # for i in states:
            #     if len(i) != 6:
            #         print("state: ", i)
            #         print("c: ", c)
            #     c += 1

            states = torch.FloatTensor(np.concatenate(states)).to(device)
            next_states = torch.FloatTensor(np.concatenate(next_states)).to(device)
            actions = torch.LongTensor(actions).to(device)
            rewards_mean = np.mean(rewards)
            # print("Episodic Reward:", np.mean(rewards))
            rewards = torch.FloatTensor(rewards).to(device)
            dones = torch.FloatTensor(dones).to(device)

            # Compute the expected Q values
            state_action_values = model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
            next_state_values = target_model(next_states).max(1)[0].detach()
            expected_state_action_values = (next_state_values * cfg.gamma * (1 - dones)) + rewards

            # Compute the loss
            loss = F.mse_loss(state_action_values, expected_state_action_values)

            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if steps_done % 1000 == 0:
                print("Episode: ", episode, " Loss: ", loss.item()," Reward: ", rewards_mean, " Epsilon: ", epsilon)

        steps_done += 1

# Setup and run the training
cfg = Config()
train(cfg)
