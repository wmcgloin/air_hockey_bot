# train_simple_Jorge.py
import pygame
import random
import numpy as np
from tensorflow.keras import models
from configuration_Jorge import patience, input_shape, num_actions, buffer_capacity, num_episodes, epsilon_start, epsilon_end, epsilon_decay, batch_size, update_target_model_steps, gamma
from Jorge_game import Game
from utility_jorge import preprocess_image
from AI_Jorge import create_model, update_model, update_target_model, select_action
from utility_jorge import ReplayBuffer

def train():
    pygame.init()
    screen = pygame.display.set_mode((800, 400))
    game = Game(screen)  # Ensure Game can accept an ai_model if needed
    buffer = ReplayBuffer(buffer_capacity)
    epsilon = epsilon_start

    total_goals = 0  # Total goals scored during training
    game_count = 0  # Count of games played

    clock = pygame.time.Clock()  # Clock to control the frame rate

    while total_goals < 14:  # Train until a certain number of goals are scored
        game.reset_game()
        state = game.get_state()
        state = preprocess_image(state)
        total_reward = 0
        steps = 0

        while not game.game_over:
            action = select_action(state, game.ai_model, epsilon, num_actions)
            next_state, reward, done = game.step(action)  # Step now also updates and renders
            next_state = preprocess_image(next_state)

            buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            # Update total goals based on rewards
            if reward != 0:
                total_goals += 1

            if len(buffer) > batch_size:
                batch = buffer.sample(batch_size)
                update_model(game.ai_model, game.ai_model, batch, gamma)

            if steps % update_target_model_steps == 0:
                update_target_model(game.ai_model, game.ai_model)

            steps += 1
            clock.tick(10)  # Slow down the training to 10 FPS for visualization

            if done:
                game_count += 1
                print(f"Game {game_count}: Total Reward: {total_reward}, Total Goals: {total_goals}")
                break

        epsilon = max(epsilon_end, epsilon * epsilon_decay)

    pygame.quit()
    print("Training completed.")
    return game.ai_model

if __name__ == "__main__":
    trained_model = train()
    trained_model.save("final_model.h5")
    print("Model saved.")
