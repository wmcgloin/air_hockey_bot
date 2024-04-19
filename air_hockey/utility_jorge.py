import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

def preprocess_image(image):
    # Convert image to grayscale and resize
    from PIL import Image
    image = Image.fromarray(image, 'RGB').convert('L')
    image = image.resize((84, 84))  # Example size, adjust as necessary
    return np.array(image)
