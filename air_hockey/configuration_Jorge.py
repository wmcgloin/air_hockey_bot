# configuration_Jorge.py
# Configuration settings for the AI in the Air Hockey game

# Neural Network Hyperparameters
learning_rate = 0.00025
gamma = 0.99  # Discount factor for future rewards
input_shape = (84, 84, 1)  # Adjust based on your preprocessed image size
num_actions = 9  # Assuming 8 directions + stay

# Training Hyperparameters
num_episodes = 1
epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay = 1
batch_size = 32
buffer_capacity = 10000
update_target_model_steps = 100  # How often to update the target model
patience = 10
