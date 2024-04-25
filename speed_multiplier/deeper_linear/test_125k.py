import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from air_hock_env import AirHockeyEnv

    
# Linear Neural network model
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.linear1 = nn.Linear(input_dim, 16)  # First hidden layer
        self.linear2 = nn.Linear(16, 32)  # Second hidden layer
        self.linear3 = nn.Linear(32, 16)  # Third hidden layer
        self.head = nn.Linear(16, output_dim)  # Output layer to produce the final outputs

    def forward(self, x):
        x = F.relu(self.linear1(x))  # Activation function for first layer
        x = F.relu(self.linear2(x))  # Activation function for second layer
        x = F.relu(self.linear3(x))  # Activation function for third layer
        x = self.head(x)  # Output layer does not need an activation for Q-value estimation
        return x
    

def load_model(model_path, outputs=None):
    env = AirHockeyEnv(render_mode='human')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DQN(8,env.action_space.n).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def run_model_in_env(model, env):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state, _ = env.reset()
    state = torch.tensor(state).float().to(device).unsqueeze(0)
    total_reward = 0
    done = False
    while not done:
        state = torch.FloatTensor(state).to(device)
        with torch.no_grad():
            action = model(state).argmax().unsqueeze(0).unsqueeze(0)
        state, reward, done, _, info = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        total_reward += reward.item()
        env.render()
    print(f"Total reward: {total_reward}")




env = AirHockeyEnv(render_mode='human')
### SINGLE MODEL
model_path = './checkpoints/policy_net_episode_125000.pth'
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