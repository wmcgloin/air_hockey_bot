# game.py
import pygame
from puck import Puck
from paddle import Paddle
import time
import numpy as np

class AirHockeyGame:
    def __init__(self, screen, mode = 'rlve'):
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

        self.mode = mode  # Store the game mode
        if self.mode == 'pve':
            self.setup_pve()  # Set up the game for player vs environment mode
        if self.mode == 'rlve':
            self.setup_rlve()

    def reset(self):
        # Reset the game state
        self.game_over = False
        self.winner = None
        self.player1_score = 0
        self.player2_score = 0
        self.puck.reset()
        self.player1_paddle.x, self.player1_paddle.y = 200, 200
        self.player2_paddle.x, self.player2_paddle.y = 600, 200
        return self.get_state()

    def setup_pve(self):
        self.basic_ai_paddle = self.player1_paddle  # Set the AI paddle to player 1 paddle
        self.basic_ai_difficulty = 0.5  # Set the AI difficulty level (0.1 to 1.0)

    def setup_rlve(self):
        # Set up the game for reinforcement learning vs environment mode
        self.basic_ai_paddle = self.player1_paddle
        self.basic_ai_difficulty = 0.5
        self.rl_ai_paddle = self.player2_paddle

    def update_pve(self, dt):
        if self.basic_ai_paddle:
            # Simple AI that moves the paddle towards the puck's y position
            if self.puck.y > self.basic_ai_paddle.y + self.basic_ai_difficulty * 50: # creates slight reaction delay
                self.basic_ai_paddle.dy = self.basic_ai_paddle.speed
            elif self.puck.y < self.basic_ai_paddle.y - self.basic_ai_difficulty * 50: # creates slight reaction delay
                self.basic_ai_paddle.dy = -self.basic_ai_paddle.speed
            else:
                self.basic_ai_paddle.dy = 0
            self.basic_ai_paddle.update_position(dt)

    def update_rlve(self, dt, action = None):
        # update basic ai
        if self.basic_ai_paddle:
            # Simple AI that moves the paddle towards the puck's y position
            if self.puck.y > self.basic_ai_paddle.y + self.basic_ai_difficulty * 50: # creates slight reaction delay
                self.basic_ai_paddle.dy = self.basic_ai_paddle.speed
            elif self.puck.y < self.basic_ai_paddle.y - self.basic_ai_difficulty * 50: # creates slight reaction delay
                self.basic_ai_paddle.dy = -self.basic_ai_paddle.speed
            else:
                self.basic_ai_paddle.dy = 0
            self.basic_ai_paddle.update_position(dt)
        # update rl ai
        if self.rl_ai_paddle:
            if action is not None:
                # 0 - no movement, 1 - move up, 2 - move down, 3 - move left, 4 - move right, 5 - move up and right, 6 - move up and left, 7 - move down and left, 8 move down and right
                # apply action to move the paddle
                mappings = {0: (0, 0), 1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0), 5: (1, -1), 6: (-1, -1), 7: (-1, 1), 8: (1, 1)}
                dx, dy = mappings.get(action) # get the dx and dy from the action mappings
                self.rl_ai_paddle.dx = self.rl_ai_paddle.speed * dx
                self.rl_ai_paddle.dy = self.rl_ai_paddle.speed * dy
                self.rl_ai_paddle.update_position(dt)
            else:
                pass

    def handle_event(self, event):
        # Handle events, currently only checks for window quit
        if event.type == pygame.QUIT:
            return False
        return True

    def update(self, dt, action = None):
        reward = 0  # Initialize reward for this update
        if self.mode == 'pvp':
            # Update game state each frame, handling player input and moving game objects
            keys = pygame.key.get_pressed()  # Get current key states within the frame
            self.player1_paddle.dx = self.player1_paddle.speed * (keys[pygame.K_d] - keys[pygame.K_a])
            self.player1_paddle.dy = self.player1_paddle.speed * (keys[pygame.K_s] - keys[pygame.K_w])

            self.player2_paddle.dx = self.player2_paddle.speed * (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT])
            self.player2_paddle.dy = self.player2_paddle.speed * (keys[pygame.K_DOWN] - keys[pygame.K_UP])

            # Update positions based on current speeds and delta time
            self.player1_paddle.update_position(dt)
            self.player2_paddle.update_position(dt)
        elif self.mode == 'pve':
            # update basic ai
            self.update_pve(dt)

            # human input
            keys = pygame.key.get_pressed()
            self.player2_paddle.dx = self.player2_paddle.speed * (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT])
            self.player2_paddle.dy = self.player2_paddle.speed * (keys[pygame.K_DOWN] - keys[pygame.K_UP])
            self.player2_paddle.update_position(dt)
        else:
            self.update_rlve(dt, action)
            # Reward calculation
            reward += self.calculate_rewards()  # Add the calculated rewards to the total reward for this update
            return reward, self.game_over, {}

        current_time = time.time()  # Get the current time
        self.puck.move(dt)
        self.puck.check_timeout(current_time)  # Check for timeout
        self.check_goal()  # Check if a goal has been scored
        self.check_collisions()  # Check for collisions between the puck and paddles
    
    def calculate_rewards(self):
        reward = 0  # Initialize reward
        # goals
        if self.left_goal.collidepoint(self.puck.x, self.puck.y): # player 2 (RL) scores
            reward += 10 # Reward for scoring a goal 
        if self.right_goal.collidepoint(self.puck.x, self.puck.y): # player 1 scores
            reward -= 10 # Penalty for letting the puck into the goal
        # winning/losing
        if self.game_over:
            if self.winner == "Player 1":
                reward -= 100  # penalty for losing
            elif self.winner == "Player 2":
                reward += 100  # reward for winning
        # puck collision with paddles
        if self.rl_ai_paddle and self.check_paddle_collision(self.rl_ai_paddle):
            reward += 0.5  # Reward for paddle2 (rl_ai_paddle) coming into contact with the puck
        # time penalty
        reward -= 0.16 / 60  # 60 fps
        return reward

    def check_paddle_collision(self, paddle):
        # Calculate collision dynamics between the puck and a paddle
        dx = self.puck.x - paddle.x  # Calculate the difference in x-coordinates between the puck and the paddle
        dy = self.puck.y - paddle.y  # Calculate the difference in y-coordinates between the puck and the paddle
        distance = (dx**2 + dy**2)**0.5  # Calculate the distance between the puck and the paddle
        if distance < self.puck.radius + paddle.radius:  # Check if the distance is less than the sum of the puck and paddle radii
            self.puck.last_hit_time = time.time()  # Update the last hit time on collision
            return True  # Return True if a collision occurred
        return False  # Return False if no collision occurred

    def get_state(self):
        # Return the current game state as an image
        state = pygame.surfarray.array3d(self.screen)
        grayscale_state = 0.299*state[:,:,0] + 0.587*state[:,:,1] + 0.114*state[:,:,2] # convert to grayscale
        # Add a channel dimension
        grayscale_state = np.expand_dims(grayscale_state, axis=-1)
        return grayscale_state

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

    def check_goal(self): # Check if the puck has entered either goal and update scores or declare a winner
        # left goal (player 2 scores)
        if self.left_goal.collidepoint(self.puck.x, self.puck.y):
            self.player2_score += 1
            if self.player2_score >= 7:
                self.game_over = True
                self.winner = "Player 2"
            self.reset_puck()

        # right goal (player 1 scores)
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
            self.puck.dx -= 1.8 * velocity_component * dx
            self.puck.dy -= 1.8 * velocity_component * dy

