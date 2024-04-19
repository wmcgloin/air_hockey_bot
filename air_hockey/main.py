# main.py
import pygame
import sys
from game import Game
import time

def main():
    pygame.init()
    size = (800, 400)
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption('Air Hockey')

    game = Game(screen)

    last_time = time.time()

    running = True

    while running:
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if game.game_over:  # Check if the game is over
                    if event.key == pygame.K_r:  # Press 'R' to restart
                        game = Game(screen)  # Reinitialize the game
                    elif event.key == pygame.K_ESCAPE:  # Press 'Escape' to quit
                        running = False

        if not game.game_over:
            game.update(dt)
            game.draw()
        else:
            game.draw_game_over()  # Display the game over screen

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
