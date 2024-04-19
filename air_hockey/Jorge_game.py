# Jorge_game.py
import pygame
import numpy as np
import time
from puck import Puck
from paddle import Paddle
from tensorflow.keras.models import load_model
from utility_jorge import preprocess_image
from AI_Jorge import select_action

class Game:
    def __init__(self, screen, ai_model=None, num_actions=9):
        # Initialize game settings
        self.screen = screen
        self.screen_width, self.screen_height = screen.get_size()
        self.game_over = False
        self.winner = None
        mid_point = self.screen_width // 2
        self.initialize_game()

        # Create game objects
        self.puck = Puck(400, 200, 15, self.screen_width, self.screen_height)
        self.player1_paddle = Paddle(200, 200, 30, self.screen_width, self.screen_height, 0, mid_point - 10)
        self.player2_paddle = Paddle(600, 200, 30, self.screen_width, self.screen_height, mid_point + 10, self.screen_width)

        # Set up the goals
        self.goal_width = 125
        self.goal_height = 15
        self.left_goal = pygame.Rect(0, (self.screen_height - self.goal_width) // 2, self.goal_height, self.goal_width)
        self.right_goal = pygame.Rect(self.screen_width - self.goal_height, (self.screen_height - self.goal_width) // 2, self.goal_height, self.goal_width)

        # Initialize scores
        self.player1_score = 0
        self.player2_score = 0

        # AI model
        self.ai_model = ai_model
        self.num_actions = num_actions

    def initialize_game(self):
        # Initialize or reset game settings
        self.game_over = False
        self.winner = None
        mid_point = self.screen_width // 2
        self.puck = Puck(400, 200, 15, self.screen_width, self.screen_height)
        self.player1_paddle = Paddle(200, 200, 30, self.screen_width, self.screen_height, 0, mid_point - 10)
        self.player2_paddle = Paddle(600, 200, 30, self.screen_width, self.screen_height, mid_point + 10, self.screen_width)
        self.goal_width = 125
        self.goal_height = 15
        self.left_goal = pygame.Rect(0, (self.screen_height - self.goal_width) // 2, self.goal_height, self.goal_width)
        self.right_goal = pygame.Rect(self.screen_width - self.goal_height, (self.screen_height - self.goal_width) // 2, self.goal_height, self.goal_width)
        self.player1_score = 0
        self.player2_score = 0

    def reset_game(self):
        self.initialize_game() 

    def run(self):
        clock = pygame.time.Clock()
        while not self.game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            self.update()
            self.draw()
            clock.tick(60)  # Cap the frame rate

        self.draw_game_over()
        pygame.display.flip()
        time.sleep(2)  # Display the game over screen for 2 seconds before closing
        pygame.quit()

    def update(self):
        dt = 1.0 / 60
        # AI controls player1_paddle if a model is provided
        if self.ai_model:
            state = self.get_state()
            state = preprocess_image(state)
            action = select_action(state, self.ai_model, 0, self.num_actions)  # Epsilon set to 0 for full exploitation
            self.apply_action(self.player1_paddle, action)

        # Human controls player2_paddle
        keys = pygame.key.get_pressed()
        self.player2_paddle.dx = self.player2_paddle.speed * (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT])
        self.player2_paddle.dy = self.player2_paddle.speed * (keys[pygame.K_DOWN] - keys[pygame.K_UP])

        self.player1_paddle.update_position(dt)
        self.player2_paddle.update_position(dt)

        self.puck.move(dt)
        self.check_goal()
        self.check_collisions()

    def get_state(self):
        pygame.image.save(self.screen, "temp.jpg")
        state = pygame.surfarray.array3d(pygame.image.load("temp.jpg"))
        return np.transpose(state, (1, 0, 2))

    def apply_action(self, paddle, action):
        mappings = {0: (0, 0), 1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0), 5: (1, 1), 6: (-1, -1), 7: (-1, 1), 8: (1, -1)}
        dx, dy = mappings.get(action, (0, 0))
        paddle.dx = paddle.speed * dx
        paddle.dy = paddle.speed * dy

    def calculate_reward(self):
        reward = 0
        if self.left_goal.collidepoint(self.puck.x, self.puck.y):
            reward -= 10  # Penalty for player 1
        if self.right_goal.collidepoint(self.puck.x, self.puck.y):
            reward += 10  # Reward for player 1
        return reward

    def draw(self):
        self.screen.fill((0, 0, 0))
        pygame.draw.rect(self.screen, (255, 255, 255), self.left_goal)
        pygame.draw.rect(self.screen, (255, 255, 255), self.right_goal)
        self.puck.draw(self.screen)
        self.player1_paddle.draw(self.screen)
        self.player2_paddle.draw(self.screen)
        font = pygame.font.Font(None, 36)
        text = font.render(f"Player 1: {self.player1_score}  Player 2: {self.player2_score}", 1, (255, 255, 255))
        self.screen.blit(text, (10, 10))
        if self.game_over:
            self.draw_game_over()

    def check_collisions(self):
        # Check collisions for both paddles
        self.paddle_collision(self.player1_paddle)
        self.paddle_collision(self.player2_paddle)

    def paddle_collision(self, paddle):
        dx = self.puck.x - paddle.x
        dy = self.puck.y - paddle.y
        distance = (dx ** 2 + dy ** 2) ** 0.5
        if distance < self.puck.radius + paddle.radius:
            self.puck.last_hit_time = time.time()  # Update the last hit time on collision
            overlap = self.puck.radius + paddle.radius - distance
            dx /= distance
            dy /= distance
            self.puck.x += dx * overlap
            self.puck.y += dy * overlap
            relative_velocity_x = self.puck.dx - paddle.dx
            relative_velocity_y = self.puck.dy - paddle.dy
            velocity_component = (relative_velocity_x * dx + relative_velocity_y * dy)
            self.puck.dx -= 2 * velocity_component * dx
            self.puck.dy -= 2 * velocity_component * dy