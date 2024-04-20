# puck.py
import pygame
import random
import math
import time


class Puck:
    def __init__(self, x, y, radius, screen_width, screen_height):
        self.x = x
        self.y = y
        self.radius = radius
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.dx = 0
        self.dy = 0
        self.last_hit_time = None  # Initialize the last hit time
        self.launch_puck()

    def reset(self):
        # Reset the puck's position and velocity
        self.x = self.screen_width // 2
        self.y = self.screen_height // 2
        self.dx = 0
        self.dy = 0
        self.launch_puck()
        self.last_hit_time = time.time()  # Reset the hit timer upon launching

    def check_timeout(self, current_time):
        # Check if the puck has not been hit for 5 seconds
        if self.last_hit_time and (current_time - self.last_hit_time > 5):
            self.reset()  # Reset the puck if 10 seconds have passed without contact

    def launch_puck(self):
        # Randomly set the initial direction and speed of the puck
        angle = random.uniform(0, 2 * math.pi)  # Choose a random angle in radians
        initial_speed = 100  # Set a constant initial speed
        self.dx = initial_speed * math.cos(angle)  # Horizontal component of the speed
        self.dy = initial_speed * math.sin(angle)  # Vertical component of the speed

    def move(self, dt):
        # Update the position of the puck based on its velocity and time elapsed
        self.x += self.dx * dt
        self.y += self.dy * dt
        friction = 0.9999  # Apply a slight friction to slow the puck over time
        self.dx *= friction
        self.dy *= friction

        # Enforce a maximum speed to prevent the puck from moving too fast
        speed = math.sqrt(self.dx**2 + self.dy**2)
        max_speed = 650  # Define the maximum allowable speed
        if speed > max_speed:
            scale = max_speed / speed
            self.dx *= scale
            self.dy *= scale

        # Check for collisions with the boundaries and reflect the puck if necessary
        if self.x - self.radius <= 0 or self.x + self.radius >= self.screen_width:
            self.dx = -self.dx  # Reverse the horizontal direction upon hitting left or right walls
        if self.y - self.radius <= 0 or self.y + self.radius >= self.screen_height:
            self.dy = -self.dy  # Reverse the vertical direction upon hitting top or bottom walls

    def draw(self, screen):
        # Draw the puck on the screen
        pygame.draw.circle(screen, (255, 0, 0), (self.x, self.y), self.radius)
