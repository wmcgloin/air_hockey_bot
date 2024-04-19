import pygame
import random
import numpy as np
from tensorflow.keras import models
from configuration_Jorge import patience

# Local imports
from AI_Jorge import create_model, update_model, update_target_model
from configuration_Jorge import (
    input_shape, num_actions, buffer_capacity, num_episodes,
    epsilon_start, epsilon_end, epsilon_decay, batch_size, update_target_model_steps, gamma
)
from Jorge_game import Game
from utility_jorge import preprocess_image
from AI_Jorge import select_action
from utility_jorge import ReplayBuffer

# Initialize both the main model and the target model
main_model = create_model(input_shape, num_actions)
target_model = create_model(input_shape, num_actions)
update_target_model(main_model, target_model)  # Sync weights

def train():
    pygame.init()
    screen = pygame.display.set_mode((800, 400))
    game = Game(screen)
    buffer = ReplayBuffer(buffer_capacity)
    epsilon = epsilon_start

    best_reward = -float('inf')  # Initialize the best observed reward
    no_improvement_count = 0  # Counter for episodes without improvement
    patience = 10  # Number of episodes to wait without improvement before stopping

    for episode in range(num_episodes):
        game.reset_game()
        state = game.get_state()
        state = preprocess_image(state)
        total_reward = 0
        steps = 0

        while not game.game_over:
            action = select_action(state, main_model, epsilon, num_actions)
            next_state, reward, done = game.step(action)
            next_state = preprocess_image(next_state)

            buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(buffer) > batch_size:
                batch = buffer.sample(batch_size)
                update_model(main_model, target_model, batch, gamma)

            if steps % update_target_model_steps == 0:
                update_target_model(main_model, target_model)

            steps += 1

            if done:
                break

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        print(f"Episode {episode + 1}, Total Reward: {total_reward}, Best Reward: {best_reward}, Epsilon: {epsilon}")

        # Early Stopping Check
        if total_reward > best_reward:
            best_reward = total_reward
            no_improvement_count = 0  # Reset counter
        else:
            no_improvement_count += 1  # Increment counter

        if no_improvement_count >= 2:
            print(f"Stopping early after {episode + 1} episodes due to no improvement.")
            break

    pygame.quit()
    print("Training completed.")

if __name__ == "__main__":
    train()
