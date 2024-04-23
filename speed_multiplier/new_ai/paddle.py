# paddle.py
import pygame

class Paddle:
    def __init__(self, x, y, radius, screen_width, screen_height, left_boundary, right_boundary, speed_multiplier=1.0):
        # Initialize paddle properties
        self.speed_multiplier = speed_multiplier
        self.x = x  # X-coordinate of the paddle's center
        self.y = y  # Y-coordinate of the paddle's center
        self.radius = radius  # Radius of the paddle
        self.speed = 6  # Speed of the paddle in pixels per second
        self.dx = 0  # Initial horizontal velocity
        self.dy = 0  # Initial vertical velocity
        self.screen_width = screen_width  # Total width of the game screen
        self.screen_height = screen_height  # Total height of the game screen
        self.left_boundary = left_boundary  # Left boundary paddle can't cross
        self.right_boundary = right_boundary  # Right boundary paddle can't cross

    def draw(self, screen):
        # Draw the paddle on the screen as a blue circle
        pygame.draw.circle(screen, (0, 0, 255), (self.x, self.y), self.radius)

    def update_position(self):
        new_x = self.x + self.dx * self.speed_multiplier
        new_y = self.y + self.dy * self.speed_multiplier

        # Apply additional padding to paddle boundaries
        padded_left_boundary = self.left_boundary + self.radius + 10  # Add 10 pixels of padding
        padded_right_boundary = self.right_boundary - self.radius - 10

        if padded_left_boundary <= new_x <= padded_right_boundary:
            self.x = new_x
        if self.radius + 15 <= new_y <= self.screen_height - self.radius - 15:  # Add 10 pixels of padding at top and bottom
            self.y = new_y