# main.py
import pygame
import sys
import argparse
from game import Game
import time

def main(mode = 'pvp'):
    pygame.init()  # Initialize all pygame modules
    size = (800, 400)  # Define the size of the game window
    screen = pygame.display.set_mode(size)  # Create a display window
    pygame.display.set_caption('Air Hockey')  # Set the title of the window

    game = Game(screen, mode)  # Initialize the game object with the screen

    last_time = time.time()  # Record the initial time for frame rate management

    running = True  # Main game loop condition

    while running:
        current_time = time.time()  # Get current time
        dt = current_time - last_time  # Calculate the time delta between frames
        last_time = current_time  # Update last_time to the current time

        # Event handling loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False  # End the game loop if the window is closed
            elif event.type == pygame.KEYDOWN:
                if game.game_over:  # Check if the game is over
                    if event.key == pygame.K_r:  # Check if 'R' is pressed
                        game = Game(screen, mode)  # Reinitialize the game to restart
                    elif event.key == pygame.K_ESCAPE:  # Check if 'Escape' is pressed
                        running = False  # End the game loop

        # Game update and rendering
        if not game.game_over:
            game.update(dt)  # Update the game logic based on the time delta
            game.draw()  # Draw the game state to the screen
        else:
            game.draw_game_over()  # Display the game over screen when the game ends

        pygame.display.flip()  # Update the full display surface to the screen

    pygame.quit()  # Uninitialize all pygame modules
    sys.exit()  # Exit the program

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Play Air Hockey')
    parser.add_argument('--mode', type=str, default='pvp',
                        choices=['pvp', 'pve', 'rlve'], # currently we have two game modes: pvp and pve
                        help='Choose the game mode: pvp, pve, rlve')
    args = parser.parse_args()
    main(args.mode)
