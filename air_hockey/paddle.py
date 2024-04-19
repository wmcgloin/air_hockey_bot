# paddle.py
import pygame

class Paddle:
    def __init__(self, x, y, radius, screen_width, screen_height, left_boundary, right_boundary):
        self.x = x
        self.y = y
        self.radius = radius
        self.speed = 300 # Speed in pixels per second
        self.dx = 0
        self.dy = 0
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.left_boundary = left_boundary
        self.right_boundary = right_boundary

    def draw(self, screen):
        pygame.draw.circle(screen, (0, 0, 255), (self.x, self.y), self.radius)

    def update_position(self, dt):
        new_x = self.x + self.dx * dt
        new_y = self.y + self.dy * dt

        # Update the position based on boundaries
        if self.left_boundary + self.radius <= new_x <= self.right_boundary - self.radius:
            self.x = new_x
        if self.radius <= new_y <= self.screen_height - self.radius:
            self.y = new_y

        # # Optionally, reduce paddle velocity slightly each frame to simulate resistance or control limits
        # drag = 0.98  # adjust the drag coefficient as needed
        # self.dx *= drag
        # self.dy *= drag
