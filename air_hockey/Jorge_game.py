import pygame
import numpy as np
from puck import Puck
from paddle import Paddle
import time

class Game:
    def __init__(self, screen):
        # Initialize game settings
        self.screen = screen  # Reference to the display window
        self.screen_width, self.screen_height = screen.get_size()  # Get the dimensions of the screen
        self.game_over = False  # Flag to determine if the game has ended
        self.winner = None  # Variable to store the winner of the game
        mid_point = self.screen_width // 2  # Calculate the middle point of the screen width

        # Create game objects
        self.puck = Puck(400, 200, 15, self.screen_width, self.screen_height)
        self.player1_paddle = Paddle(200, 200, 30, self.screen_width, self.screen_height, 0, mid_point - 10)
        self.player2_paddle = Paddle(600, 200, 30, self.screen_width, self.screen_height, mid_point + 10, self.screen_width)

        # Set up the goals
        self.goal_width = 125  # Width of the goal area
        self.goal_height = 15  # Height of the goal, extending from top and bottom of the screen
        self.left_goal = pygame.Rect(0, (self.screen_height - self.goal_width) // 2, self.goal_height, self.goal_width)
        self.right_goal = pygame.Rect(self.screen_width - self.goal_height, (self.screen_height - self.goal_width) // 2, self.goal_height, self.goal_width)
        
        # Initialize scores
        self.player1_score = 0
        self.player2_score = 0

    def handle_event(self, event):
        # Handle events, currently only checks for window quit
        if event.type == pygame.QUIT:
            return False
        return True
    
    def update(self, dt):
        # Update game state each frame, handling player input and moving game objects
        keys = pygame.key.get_pressed()  # Get current key states within the frame
        self.player1_paddle.dx = self.player1_paddle.speed * (keys[pygame.K_d] - keys[pygame.K_a])
        self.player1_paddle.dy = self.player1_paddle.speed * (keys[pygame.K_s] - keys[pygame.K_w])

        self.player2_paddle.dx = self.player2_paddle.speed * (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT])
        self.player2_paddle.dy = self.player2_paddle.speed * (keys[pygame.K_DOWN] - keys[pygame.K_UP])

        # Update positions based on current speeds and delta time
        self.player1_paddle.update_position(dt)
        self.player2_paddle.update_position(dt)

        current_time = time.time()  # Get the current time
        self.puck.move(dt)
        self.puck.check_timeout(current_time)  # Check for timeout
        self.check_goal()  # Check if a goal has been scored
        self.check_collisions()  # Check for collisions between the puck and paddles

    def get_state(self):
        # Return the current game state as a numpy array
        # You may want to downscale or process this further
        pygame.image.save(self.screen, "temp.jpg")
        state = pygame.surfarray.array3d(pygame.image.load("temp.jpg"))
        return np.transpose(state, (1, 0, 2))

    def step(self, action):
        # Apply action to the game
        # 'action' could be a tuple (action_p1, action_p2) with each being an integer
        self.apply_action(self.player1_paddle, action[0])
        self.apply_action(self.player2_paddle, action[1])
        
        # Update game logic
        dt = 1.0 / 60  # Assuming 60 FPS
        self.update(dt)
        self.draw()

        # Calculate reward
        reward = self.calculate_reward()

        # Check if the game has ended
        done = self.game_over

        # Get the next state
        next_state = self.get_state()

        return next_state, reward, done

    def apply_action(self, paddle, action):
        # Update paddle direction based on action
        # Example: 0 = Stay, 1 = Up, 2 = Down, 3 = Left, 4 = Right
        mappings = {0: (0, 0), 1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0), 5: (1, 1), 6: (-1, -1), 7: (-1, 1), 8: (1, -1)}
        dx, dy = mappings.get(action, (0, 0))
        paddle.dx = paddle.speed * dx
        paddle.dy = paddle.speed * dy

    def calculate_reward(self):
        # Example reward function
        reward = 0
        if self.left_goal.collidepoint(self.puck.x, self.puck.y):
            reward -= 10  # Penalty for player 1
        if self.right_goal.collidepoint(self.puck.x, self.puck.y):
            reward += 10  # Reward for player 1
        return reward
    
    def reset_game(self):
        # Reset all game settings to the initial state
        self.puck.reset()
        #self.player1_paddle.reset(200, 200)  # Assuming reset method exists for paddles
        #self.player2_paddle.reset(600, 200)
        self.player1_score = 0
        self.player2_score = 0
        self.game_over = False
        self.winner = None

    def draw_game_over(self):
        # Render game over screen with winner information and instructions
        self.screen.fill((0, 0, 0))
        font = pygame.font.Font(None, 74)
        text = font.render(f"Game Over - {self.winner} Wins!", True, (255, 255, 255))
        text_rect = text.get_rect(center=(self.screen_width / 2, self.screen_height / 2 - 50))
        self.screen.blit(text, text_rect)
        instructions_font = pygame.font.Font(None, 48)
        restart_text = instructions_font.render("Press 'R' to Restart", True, (255, 255, 255))
        restart_rect = restart_text.get_rect(center=(self.screen_width / 2, self.screen_height / 2 + 20))
        self.screen.blit(restart_text, restart_rect)
        quit_text = instructions_font.render("Press 'Esc' to Quit", True, (255, 255, 255))
        quit_rect = quit_text.get_rect(center=(self.screen_width / 2, self.screen_height / 2 + 70))
        self.screen.blit(quit_text, quit_rect)

    def check_goal(self):
        # Check if the puck has entered either goal and update scores or declare a winner
        if self.left_goal.collidepoint(self.puck.x, self.puck.y):
            self.player2_score += 1
            if self.player2_score >= 7:
                self.game_over = True
                self.winner = "Player 2"
            self.reset_puck()

        elif self.right_goal.collidepoint(self.puck.x, self.puck.y):
            self.player1_score += 1
            if self.player1_score >= 7:
                self.game_over = True
                self.winner = "Player 1"
            self.reset_puck()

    def reset_puck(self):
        # Reset the puck position and direction after a goal is scored
        self.puck.x, self.puck.y = self.screen_width // 2, self.screen_height // 2
        self.puck.dx, self.puck.dy = 0, 0
        self.puck.launch_puck()

    def draw(self):
        # Draw all game elements to the screen
        if not self.game_over:
            self.screen.fill((0, 0, 0))
            pygame.draw.rect(self.screen, (255, 255, 255), self.left_goal)
            pygame.draw.rect(self.screen, (255, 255, 255), self.right_goal)
            self.puck.draw(self.screen)
            self.player1_paddle.draw(self.screen)
            self.player2_paddle.draw(self.screen)
            font = pygame.font.Font(None, 36)
            text = font.render(f"Player 1: {self.player1_score}", 1, (255, 255, 255))
            self.screen.blit(text, (50, 20))
            text = font.render(f"Player 2: {self.player2_score}", 1, (255, 255, 255))
            self.screen.blit(text, (self.screen_width - 150, 20))
        else:
            self.draw_game_over()  # Show game over screen when the game has ended

    def check_collisions(self):
        # Manage collisions between the puck and paddles
        self.paddle_collision(self.player1_paddle)
        self.paddle_collision(self.player2_paddle)

    def paddle_collision(self, paddle):
        # Calculate collision dynamics between a puck and a paddle
        dx = self.puck.x - paddle.x
        dy = self.puck.y - paddle.y
        distance = (dx**2 + dy**2)**0.5
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