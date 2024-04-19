# game.py
import pygame
from puck import Puck
from paddle import Paddle

class Game:
    def __init__(self, screen):
        self.screen = screen
        self.screen_width, self.screen_height = screen.get_size()
        self.game_over = False  # Initially the game is not over
        self.winner = None  # Variable to keep track of who won
        mid_point = self.screen_width // 2
        self.puck = Puck(400, 200, 15, self.screen_width, self.screen_height)
        self.player1_paddle = Paddle(200, 200, 30, self.screen_width, self.screen_height, 0, mid_point - 10)
        self.player2_paddle = Paddle(600, 200, 30, self.screen_width, self.screen_height, mid_point + 10, self.screen_width)
        self.goal_width = 125  # Width of the goal area
        self.goal_height = 15  # Height of the goal, extending from top and bottom of the screen
        self.left_goal = pygame.Rect(0, (self.screen_height - self.goal_width) // 2, self.goal_height, self.goal_width)
        self.right_goal = pygame.Rect(self.screen_width - self.goal_height, (self.screen_height - self.goal_width) // 2, self.goal_height, self.goal_width)
        
        # Score initialization
        self.player1_score = 0
        self.player2_score = 0
        self.game_over = False

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            return False
        return True

    def update(self, dt):
        keys = pygame.key.get_pressed()  # Get current key states within the frame
        self.player1_paddle.dx = self.player1_paddle.speed * (keys[pygame.K_d] - keys[pygame.K_a])
        self.player1_paddle.dy = self.player1_paddle.speed * (keys[pygame.K_s] - keys[pygame.K_w])

        self.player2_paddle.dx = self.player2_paddle.speed * (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT])
        self.player2_paddle.dy = self.player2_paddle.speed * (keys[pygame.K_DOWN] - keys[pygame.K_UP])

        self.player1_paddle.update_position(dt)
        self.player2_paddle.update_position(dt)

        self.puck.move(dt)
        self.check_goal()
        self.check_collisions()

    def draw_game_over(self):
        # Fill the screen with a dark overlay
        self.screen.fill((0, 0, 0))

        # Prepare the game over text
        font = pygame.font.Font(None, 74)
        text = font.render(f"Game Over - {self.winner} Wins!", True, (255, 255, 255))
        text_rect = text.get_rect(center=(self.screen_width / 2, self.screen_height / 2 - 50))
        self.screen.blit(text, text_rect)

        # Prepare the restart and quit instructions
        instructions_font = pygame.font.Font(None, 48)
        restart_text = instructions_font.render("Press 'R' to Restart", True, (255, 255, 255))
        restart_rect = restart_text.get_rect(center=(self.screen_width / 2, self.screen_height / 2 + 20))
        self.screen.blit(restart_text, restart_rect)

        quit_text = instructions_font.render("Press 'Esc' to Quit", True, (255, 255, 255))
        quit_rect = quit_text.get_rect(center=(self.screen_width / 2, self.screen_height / 2 + 70))
        self.screen.blit(quit_text, quit_rect)

    def check_goal(self):
        # Check for scoring in the left goal
        if self.left_goal.collidepoint(self.puck.x, self.puck.y):
            self.player2_score += 1
            if self.player2_score >= 7:
                self.game_over = True
                self.winner = "Player 2"
            self.reset_puck()

        # Check for scoring in the right goal
        elif self.right_goal.collidepoint(self.puck.x, self.puck.y):
            self.player1_score += 1
            if self.player1_score >= 7:
                self.game_over = True
                self.winner = "Player 1"
            self.reset_puck()

    def reset_puck(self):
        # Reset the puck to the center of the screen with a random direction
        self.puck.x, self.puck.y = self.screen_width // 2, self.screen_height // 2
        self.puck.dx, self.puck.dy = 0, 0  # Stop the puck movement before re-launching
        self.puck.launch_puck()  # Reinitialize the puck's direction

    def draw(self):
        if not self.game_over:
            self.screen.fill((0, 0, 0))  # Clear the screen
            pygame.draw.rect(self.screen, (255, 255, 255), self.left_goal)  # Draw left goal
            pygame.draw.rect(self.screen, (255, 255, 255), self.right_goal)  # Draw right goal
            self.puck.draw(self.screen)
            self.player1_paddle.draw(self.screen)
            self.player2_paddle.draw(self.screen)
        
            # Display scores
            font = pygame.font.Font(None, 36)
            text = font.render(f"Player 1: {self.player1_score}", 1, (255, 255, 255))
            self.screen.blit(text, (50, 20))
            text = font.render(f"Player 2: {self.player2_score}", 1, (255, 255, 255))
            self.screen.blit(text, (self.screen_width - 150, 20))
        
        else:
            # Display game over and the winner
            self.screen.fill((0, 0, 0))  # Optionally clear the screen or draw a game over background
            font = pygame.font.Font(None, 74)
            text = font.render(f"Game Over - {self.winner} Wins!", 1, (255, 255, 255))
            text_rect = text.get_rect(center=(self.screen_width/2, self.screen_height/2))
            self.screen.blit(text, text_rect)

    def check_collisions(self):
        # Check collision between the puck and each paddle
        self.paddle_collision(self.player1_paddle)
        self.paddle_collision(self.player2_paddle)

    def paddle_collision(self, paddle):
        # Calculate the distance and direction vector between the puck and the paddle
        dx = self.puck.x - paddle.x
        dy = self.puck.y - paddle.y
        distance = (dx**2 + dy**2)**0.5

        # Check if the distance is less than the sum of their radii
        if distance < self.puck.radius + paddle.radius:
            overlap = self.puck.radius + paddle.radius - distance
            # Normalize the distance vector
            dx /= distance
            dy /= distance

            # Reposition the puck outside the paddle to avoid sticking
            self.puck.x += dx * overlap
            self.puck.y += dy * overlap

            # Reflect the puck's direction based on the paddle's movement
            # Using a simple model where the puck's new velocity is a combination of its old velocity
            # and twice the component of the paddle's velocity in the direction of the normalized vector
            relative_velocity_x = self.puck.dx - paddle.dx
            relative_velocity_y = self.puck.dy - paddle.dy
            velocity_component = (relative_velocity_x * dx + relative_velocity_y * dy)
            self.puck.dx -= 2 * velocity_component * dx
            self.puck.dy -= 2 * velocity_component * dy
