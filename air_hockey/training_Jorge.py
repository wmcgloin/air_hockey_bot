# utility_jorge.py

from AI_Jorge import create_model, update_model, ReplayBuffer, update_target_model
from configuration_Jorge import input_shape, num_actions, buffer_capacity, num_episodes, epsilon_start, epsilon_end, epsilon_decay, batch_size, update_target_model_steps
from Jorge_game import Game
from utility_jorge import preprocess_image
import pygame
import random  # Make sure to import random if not already done
from configuration_Jorge import gamma

# Define both the main model and the target model
main_model = create_model(input_shape, num_actions)
target_model = create_model(input_shape, num_actions)

update_target_model(main_model, target_model)  # Ensure target model is initialized with the same weights as main model


def train():
    pygame.init()
    screen = pygame.display.set_mode((800, 400))  # Define screen size
    game = Game(screen)

    # No need to define 'model' here again since 'main_model' and 'target_model' are already defined
    buffer = ReplayBuffer(buffer_capacity)

    epsilon = epsilon_start
    steps = 0

    for episode in range(num_episodes):
        game.reset_game()
        state = game.get_state()
        state = preprocess_image(state)
        total_reward = 0

        while not game.game_over:
            action = select_action(state, main_model, epsilon, num_actions)  # Use main_model here
            next_state, reward, done = game.step(action)
            next_state = preprocess_image(next_state)

            buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

            if len(buffer) > batch_size:
                batch = buffer.sample(batch_size)
                update_model(main_model, target_model, batch, gamma)  # Ensure both models are used

            if steps % update_target_model_steps == 0:
                update_target_model(main_model, target_model)  # Correctly update the target model

            steps += 1

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        print(f'Episode {episode + 1}, Total Reward: {total_reward}, Epsilon: {epsilon}')


def select_action(state, model, epsilon, num_actions):
    if random.random() < epsilon:
        return (random.randint(0, num_actions - 1), random.randint(0, num_actions - 1))
    else:
        q_values = model.predict(state[np.newaxis, :])
        # Assuming the model predicts actions for both paddles, split the output
        action1 = np.argmax(q_values[0][:num_actions//2])
        action2 = np.argmax(q_values[0][num_actions//2:])
        return (action1, action2)


if __name__ == "__main__":
    train()