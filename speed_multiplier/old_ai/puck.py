# puck.py
import pygame
import random
import math

class Puck:
    def __init__(self, x, y, radius, screen_width, screen_height, speed_multiplier=1.0):
        self.speed_multiplier = speed_multiplier
        self.x = x
        self.y = y
        self.radius = radius
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.dx = 0
        self.dy = 0
        self.launch_puck()

    def reset(self):
        # Reset the puck's position and velocity
        self.x = self.screen_width // 2
        self.y = self.screen_height // 2
        self.dx = 0
        self.dy = 0
        self.launch_puck()

    def launch_puck(self):
        # Randomly set the initial direction and speed of the puck
        angle = random.uniform(0, 2 * math.pi)  # Choose a random angle in radians
        initial_speed = 12  # Set a constant initial speed
        self.dx = initial_speed * math.cos(angle)  # Horizontal component of the speed
        self.dy = initial_speed * math.sin(angle)  # Vertical component of the speed

    def move(self):
        # Update the position of the puck based on its velocity
        self.x += self.dx
        self.y += self.dy

        # # Enforce a maximum speed to prevent the puck from moving too fast
        # speed = math.sqrt(self.dx**2 + self.dy**2)
        # max_speed = 20  # Define the maximum allowable speed
        # if speed > max_speed:
        #     scale = max_speed / speed
        #     self.dx *= scale
        #     self.dy *= scale

        # if self.x > max_speed:
        #     self.x = max_speed
        # if self.y > max_speed:
        #     self.y = max_speed

        # Set max speed for the puck to be moving at
        max_speed = 24

        # Check if the puck is moving too fast
        if self.dx > max_speed:
            self.dx = max_speed
        if self.dy > max_speed:
            self.dy = max_speed



        # Check for collisions with the boundaries and reflect the puck if necessary
        if self.x - self.radius <= 0 or self.x + self.radius >= self.screen_width:
            self.dx = -self.dx  # Reverse the horizontal direction upon hitting left or right walls
        if self.y - self.radius <= 0 or self.y + self.radius >= self.screen_height:
            self.dy = -self.dy  # Reverse the vertical direction upon hitting top or bottom walls

    def draw(self, screen):
        # Draw the puck on the screen
        pygame.draw.circle(screen, (255, 0, 0), (self.x, self.y), self.radius)
