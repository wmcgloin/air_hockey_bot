# puck.py
import pygame
import random
import math


import pygame
import random
import math

class Puck:
    def __init__(self, x, y, radius, screen_width, screen_height):
        self.x = x
        self.y = y
        self.radius = radius
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.dx = 0  # Initialize with no movement
        self.dy = 0
        self.launch_puck()  # Set initial direction

    def launch_puck(self):
        angle = random.uniform(0, 2 * math.pi)  # Random angle in radians
        initial_speed = 100  # Define a consistent initial speed
        self.dx = initial_speed * math.cos(angle)  # Calculate x component
        self.dy = initial_speed * math.sin(angle)  # Calculate y component

    def move(self, dt):
        # Move the puck and apply friction
        self.x += self.dx * dt
        self.y += self.dy * dt
        friction = 0.99999999  # Mild friction
        self.dx *= friction
        self.dy *= friction

        # Enforce maximum speed
        speed = math.sqrt(self.dx**2 + self.dy**2)
        max_speed = 800
        if speed > max_speed:
            scale = max_speed / speed
            self.dx *= scale
            self.dy *= scale

        # Boundary checking and reverse direction if hitting a wall
        if self.x - self.radius <= 0 or self.x + self.radius >= self.screen_width:
            self.dx = -self.dx
        if self.y - self.radius <= 0 or self.y + self.radius >= self.screen_height:
            self.dy = -self.dy

    def draw(self, screen):
        pygame.draw.circle(screen, (255, 0, 0), (self.x, self.y), self.radius)