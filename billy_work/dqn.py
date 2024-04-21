# dqn.py
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
from collections import namedtuple

# Assuming the environment is defined as shown in your code.
from air_hock_env import AirHockeyEnv

# hyperparameters
BATCH_SIZE = 64 
GAMMA = 0.99  # Discount factor for future rewards
EPS_START = 0.95  # Initial epsilon value for epsilon-greedy action selection
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

# class designed to hold a collection of previous experiences or transitions the agent 
# has observed. This allows the reinforcement learning agent to sample from past experiences 
# and learn from them, avoiding the need to learn solely from immediate real-time interactions.
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)  # Initialize a double-ended queue with a fixed maximum length

    def push(self, *args):
        """Save a transition."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)  # Randomly sample a batch of transitions from the memory

    def __len__(self):
        return len(self.memory)  # Return the current size of the memory

# Neural network model
class DQN(nn.Module):
    def __init__(self, outputs):
        super(DQN, self).__init__()
        # Change the first convolutional layer to accept a single channel input
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)  # Only 1 input channel
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.AvgPool2d(2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool3 = nn.AvgPool2d(2)
        
        # Function to automatically determine the features for the linear layer
        self._to_linear = None
        self._get_conv_output([1, 1, 800, 400])  # Update to single channel input for feature size calculation

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

# Instantiate the policy and target networks from the DQN class defined earlier
policy_net = DQN(env.action_space.n).to(device)  # Initialize policy network for action value estimation
target_net = DQN(env.action_space.n).to(device)  # Initialize target network to stabilize learning
target_net.load_state_dict(policy_net.state_dict())  # Copy weights from policy_net to target_net
target_net.eval()  # Set target_net to evaluation mode, which disables training specific layers like dropout

# Set up the optimizer for the policy network
optimizer = optim.Adam(policy_net.parameters(), lr=LR)  # Use the Adam optimizer for policy_net with a specified learning rate

# Initialize the replay memory
memory = ReplayMemory(MEMORY_CAPACITY)  # Create a ReplayMemory object with a defined capacity

# Define a function to select an action based on the current state and the number of steps done
def select_action(state, steps_done):
    sample = random.random()  # Generate a random sample for epsilon-greedy strategy
    # Calculate epsilon threshold for epsilon-greedy action selection
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():  # Temporarily set all the requires_grad flag to false
            # Select action with the highest predicted Q-value from policy_net
            return policy_net(state).max(1)[1].view(1, 1)  # Get the action with the highest Q-value
    else:
        # Return a random action within the action space of the environment
        return torch.tensor([[random.randrange(env.action_space.n)]], device=device, dtype=torch.long)

# function iteratively improves policy network by minimizing the difference between predicted Q-values 
# and the target Q-values derived from the Bellman equation, thus refining the agent's policy to maximize future rewards

def optimize_model():
    if len(memory) < BATCH_SIZE:

        return

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Prepare the tensors from the sampled transitions
    non_final_mask = torch.tensor([s is not None for s in batch.next_state], device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(device)
    state_batch = torch.cat([s for s in batch.state]).to(device)
    action_batch = torch.cat([s for s in batch.action]).to(device)
    reward_batch = torch.cat([s for s in batch.reward]).to(device)

    print(f"Shape of state_batch: {state_batch.shape}")
    print(f"Shape of action_batch: {action_batch.shape}")

    # Network forward pass
    state_action_values = policy_net(state_batch)

    # Ensure action_batch is correctly shaped and of the correct dtype for gather
    action_batch = action_batch.long().view(-1, 1)
    print(f"Shape of action_batch for gather: {action_batch.shape}")

    # Using gather to select the Q-values corresponding to the taken actions
    state_action_values = state_action_values.gather(1, action_batch)

    # Compute V(s_{t+1}) for all non-final next states using the target network
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    if non_final_next_states.size(0) > 0:
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Backpropagation
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
    state = torch.from_numpy(state).unsqueeze(0).unsqueeze(0).float().to(device)  # Ensure state is correctly shaped
    total_reward = 0

    for t in count():
        action = select_action(state, steps_done)
        next_state, reward, done, truncated, info = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        
        if not done:
            next_state = torch.from_numpy(next_state).unsqueeze(0).unsqueeze(0).float().to(device)  # Same reshaping for next_state
        else:
            next_state = None

        memory.push(state, action, next_state, reward)
        state = next_state if next_state is not None else state  # Ensure continuity in states

        optimize_model()
        if done:
            episode_rewards.append(total_reward)
            plot_rewards()
            break

    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
plt.ioff()  # Turn off interactive mode
plt.show()  # Display results