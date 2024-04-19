# play_against_ai_Jorge.py 

import pygame
from tensorflow.keras.models import load_model
from Jorge_game import Game
from utility_jorge import preprocess_image
from AI_Jorge import select_action

def play_against_ai():
    pygame.init()
    screen = pygame.display.set_mode((800, 400))
    ai_model = load_model("final_model.h5")
    game = Game(screen, ai_model=ai_model)

    done = False
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        state = game.get_state()
        state = preprocess_image(state)
        action = select_action(state, ai_model, 0, num_actions)  # AI plays with no exploration
        game.step(action)  # Apply the AI's action

        pygame.display.flip()  # Update the screen

    pygame.quit()
    print("Game Over!")

if __name__ == "__main__":
    play_against_ai()
