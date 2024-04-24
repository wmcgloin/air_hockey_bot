import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
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
    

def load_model(model_path, outputs=None):
    env = PongV2(mode='test')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DQN(env.observation_space.shape[0],env.action_space.n).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def run_model_in_env(model, env):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # env.render(mode='human')
    state = env.reset()
    state = np.reshape(state, [1, env.observation_space.shape[0]])
    total_reward = 0
    done = False
    while not done:
        # print("State: ")
        # print(state)
        # print("State shape: ")
        # print(state.shape)
        state = torch.FloatTensor(state).to(device)
        with torch.no_grad():
            action = model(state).argmax()
        state, reward, done, _ = env.step(action.item())
        total_reward += reward
        env.render()
    print(f"Total reward: {total_reward}")




env = PongV2(mode='test')
### SINGLE MODEL
model_path = './checkpoints/policy_net_episode_150000.pth'
while True:
    model = load_model(model_path)
    run_model_in_env(model, env)
    # input("Press Enter to continue...")
    # done = False
    # total_reward = 0
    # while not done:
    #     state = torch.tensor(state).float().to(device)
    #     q_values = model(state)
    #     action = torch.argmax(q_values).item()
    #     state, reward, done, _ = env.step(action)
    #     total_reward += reward
    # return total_reward


### ALL MODELS
checkpoint_dir = './checkpoints'
model_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
for model_file in model_files:
    model_path = os.path.join(checkpoint_dir, model_file)
    model = load_model(model_path)
    print(f'Running model {model_file} in environment')
    run_model_in_env(model, env)

env.close()
    # done = False
    # total_reward = 0
    # while not done:
    #     state = torch.tensor(state).float().to(device)
    #     q_values = model(state)
    #     action = torch.argmax(q_values).item()
    #     state, reward, done, _ = env.step(action)
    #     total_reward += reward
    # return total_reward