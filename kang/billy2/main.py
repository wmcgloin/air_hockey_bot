# main.py
import pygame
import sys
import argparse
from game import AirHockeyGame
import time

def main(mode = 'pvp', operation_mode = 'realtime'):
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

    pygame.quit()  # Uninitialize all pygame modules
    sys.exit()  # Exit the program

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Play Air Hockey')
    parser.add_argument('--mode', type=str, default='pvp',
                        choices=['pvp', 'pve', 'rlve', 'random'], # currently we have two game modes: pvp and pve
                        help='Choose the game mode: pvp, pve, rlve, random')
    args = parser.parse_args()
    main(args.mode)


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