import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import math
from collections import namedtuple, deque
from itertools import count
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

# Assuming the environment is defined as shown in your code.
from air_hock_env import AirHockeyEnv

# hyperparameters
BATCH_SIZE = 64 
GAMMA = 0.99  # Discount factor for future rewards
EPS_START = 0.9  # Initial epsilon value for epsilon-greedy action selection
EPS_END = 0.05  # Final epsilon value for epsilon-greedy action selection
EPS_DECAY = 200  # Rate at which epsilon decreases
LR = 1e-4  # Learning rate for the optimizer
TARGET_UPDATE = 10  # How often to update the target network
MEMORY_CAPACITY = 10000  # Capacity of the replay memory
NUM_EPISODES = 50  # Number of episodes to train

# Set up environment directly using your custom class
env = AirHockeyEnv()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Replay memory to store transitions
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def push(self, *args):
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

# Neural network model
class DQN(nn.Module):
    def __init__(self, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.AvgPool2d(2)  # Average pooling layer after first conv layer
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.AvgPool2d(2)  # Average pooling layer after second conv layer
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool3 = nn.AvgPool2d(2)  # Average pooling layer after third conv layer
        
        # Calculate the size of the output from the final pooling layer
        self._to_linear = None
        self._get_conv_output([1, 3, 800, 400])

        self.head = nn.Linear(self._to_linear, outputs)

    def _get_conv_output(self, shape):
        input = torch.rand(shape)
        output = self.pool1(self.bn1(self.conv1(input)))
        output = self.pool2(self.bn2(self.conv2(output)))
        output = self.pool3(self.bn3(self.conv3(output)))
        self._to_linear = int(torch.numel(output) / output.shape[0])

    def forward(self, x):
        x = F.relu(self.pool1(self.bn1(self.conv1(x))))
        x = F.relu(self.pool2(self.bn2(self.conv2(x))))
        x = F.relu(self.pool3(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        return self.head(x)

# Assuming the outputs variable is defined (number of actions in your environment)
# outputs = env.action_space.n (make sure to define this before initializing DQN if it's not already defined)
# model = DQN(outputs)


policy_net = DQN(env.action_space.n).to(device)
target_net = DQN(env.action_space.n).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayMemory(MEMORY_CAPACITY)

def select_action(state, steps_done):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(env.action_space.n)]], device=device, dtype=torch.long)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    # Add this code before the concatenation operation
    for i, state in enumerate(batch.state):
        print(f"Tensor {i} shape: {state.shape}")
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Set up plotting
plt.ion()  # Interactive mode on
fig, ax = plt.subplots()
episode_rewards = []

def plot_rewards():
    ax.clear()
    ax.plot(episode_rewards)
    ax.set_title('Episode vs Rewards')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative Reward')
    plt.draw()
    plt.pause(0.001)  # pause to update plots

# Training loop
steps_done = 0
for i_episode in range(NUM_EPISODES):
    env.reset()
    state = env.get_state()
    state = torch.from_numpy(state).permute(2, 0, 1).unsqueeze(0).float().to(device)
    total_reward = 0

    for t in count():
        action = select_action(state, steps_done)
        next_state, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        total_reward += reward.item()

        if not done:
            next_state = torch.from_numpy(next_state).permute(2, 0, 1).unsqueeze(0).float().to(device)
        else:
            next_state = None

        memory.push(state, action, next_state, reward)
        state = next_state

        optimize_model()
        if done:
            episode_rewards.append(total_reward)
            plot_rewards()
            break

    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
plt.ioff()  # Turn off interactive mode
plt.show()  # Keep the window open at the end of the training