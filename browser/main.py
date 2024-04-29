# import os
# import torch
# import numpy as np
# import torch.nn as nn
# import torch.nn.functional as F
# from air_hock_env import AirHockeyEnv
import asyncio

    

# # Linear Neural network model
# class DQN(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(DQN, self).__init__()
#         self.linear1 = nn.Linear(input_dim, 16)  # First hidden layer
#         self.linear2 = nn.Linear(16, 32)  # Second hidden layer
#         self.linear3 = nn.Linear(32, 16)  # Third hidden layer
#         self.head = nn.Linear(16, output_dim)  # Output layer to produce the final outputs

#     def forward(self, x):
#         x = F.relu(self.linear1(x))  # Activation function for first layer
#         x = F.relu(self.linear2(x))  # Activation function for second layer
#         x = F.relu(self.linear3(x))  # Activation function for third layer
#         x = self.head(x)  # Output layer does not need an activation for Q-value estimation
#         return x
    

# def load_model(model_path, outputs=None):
#     env = AirHockeyEnv(render_mode='human')
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = DQN(8,env.action_space.n).to(device)
#     model.load_state_dict(torch.load(model_path))
#     model.eval()
#     return model

# def run_model_in_env(model, env):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     state, _ = env.reset()
#     state = torch.tensor(state).float().to(device).unsqueeze(0)
#     total_reward = 0
#     done = False
#     while not done:
#         state = torch.FloatTensor(state).to(device)
#         with torch.no_grad():
#             action = model(state).argmax().unsqueeze(0).unsqueeze(0)
#         state, reward, done, _, info = env.step(action.item())
#         reward = torch.tensor([reward], device=device)
#         total_reward += reward.item()
#         env.render()
#     print(f"Total reward: {total_reward}")




# env = AirHockeyEnv(render_mode='human',mode='play')
# ### SINGLE MODEL
# model_path = 'checkpoints//policy_net_episode_300000.pth'
# model = load_model(model_path)

# async def main():
#     while True:
#         global model, env
#         run_model_in_env(model, env)
#         await asyncio.sleep(0)  # Let other tasks run


# # This is the program entry point
# asyncio.run(main())

# main.py
import pygame
import sys
import argparse
from game import AirHockeyGame
import time

async def main(mode = 'pvp', operation_mode = 'realtime'):
    pygame.init()  # Initialize all pygame modules
    size = (800, 400)  # Define the size of the game window
    screen = pygame.display.set_mode(size)  # Create a display window
    pygame.display.set_caption('Air Hockey')  # Set the title of the window

    game = AirHockeyGame(screen, mode)  # Initialize the game object with the screen
    clock = pygame.time.Clock()  # Initialize clock object for frame rate control    

    running = True  # Main game loop condition
    tick_rate = 60 if operation_mode == 'realtime' else 1000  # For simulation, process many ticks per frame


    while running:
        if operation_mode == 'realtime':
            clock.tick(tick_rate)  # Limit the frame rate in real-time mode
        else:
            pygame.event.pump()  # Handle events without waiting

        # Event handling loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False  # End the game loop if the window is closed
            elif event.type == pygame.KEYDOWN:
                if game.game_over:  # Check if the game is over
                    if event.key == pygame.K_r:  # Check if 'R' is pressed
                        game = AirHockeyGame(screen, mode)  # Reinitialize the game to restart
                    elif event.key == pygame.K_ESCAPE:  # Check if 'Escape' is pressed
                        running = False  # End the game loop

        # Game update and rendering
        if not game.game_over:
            game.update(tick_rate)  # Update the game logic based on the time delta
            game.draw()  # Draw the game state to the screen
        else:
            game.draw_game_over()  # Display the game over screen when the game ends

        pygame.display.flip()  # Update the full display surface to the screen
        await asyncio.sleep(0)  # Let other tasks run

    pygame.quit()  # Uninitialize all pygame modules
    sys.exit()  # Exit the program

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Play Air Hockey')
    parser.add_argument('--mode', type=str, default='pvp',
                        choices=['pvp', 'pve', 'rlve', 'random'], # currently we have two game modes: pvp and pve
                        help='Choose the game mode: pvp, pve, rlve, random')
    args = parser.parse_args()
    main(args.mode)
    asyncio.run(main())



# def main(mode='pvp', operation_mode='realtime'):
#     pygame.init()
#     size = (800, 400)
#     screen = pygame.display.set_mode(size)
#     pygame.display.set_caption('Air Hockey')

#     game = AirHockeyGame(screen, mode)
#     clock = pygame.time.Clock()

#     running = True
#     tick_rate = 60 if operation_mode == 'realtime' else 1000  # For simulation, process many ticks per frame

#     while running:
#         if operation_mode == 'realtime':
#             clock.tick(tick_rate)  # Limit the frame rate in real-time mode
#         else:
#             pygame.event.pump()  # Handle events without waiting

#         # Process events
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 running = False
#             elif event.type == pygame.KEYDOWN:
#                 if game.game_over:
#                     running = False

#         # Update game state
#         for _ in range(tick_rate):
#             game.update()  # Update game based on ticks

#         # Render the current game state
#         if operation_mode == 'realtime':
#             game.render()
#             pygame.display.flip()

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Run the Air Hockey game.')
#     parser.add_argument('--mode', type=str, default='pvp', choices=['pvp', 'pve', 'simulation'],
#                         help='Game mode (pvp, pve, or simulation for AI training)')
#     parser.add_argument('--operation_mode', type=str, default='realtime', choices=['realtime', 'simulation'],
#                         help='Operation mode of the game (realtime for human players, simulation for AI training)')
#     args = parser.parse_args()
#     main(args.mode, args.operation_mode)