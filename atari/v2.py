import pygame
import random
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import os

# Constants should be in UPPER_CASE
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
STATE_SIZE = 4
ACTION_SIZE = 3
LEARNING_RATE = 0.001
DISCOUNT_RATE = 0.99
BATCH_SIZE = 32
MEMORY_SIZE = 10000
BALL_SPEED = 10
RND_DIR_LIST = (-1, -0.8, -0.6, 0.6, 0.8, 1)
PADDLE_SPEED = 10
PADDLE_WIDTH = 140
PADDLE_HEIGHT = 10
BALL_SIZE = 20
BRICK_WIDTH = 40
BRICK_HEIGHT = 20
BRICK_PADDING = 5
REWARD_PADDLE_HIT = 3
REWARD_BRICK_HIT = 3
REWARD_GAME_OVER = -100
REWARD_GAME_WIN = 100

COLORS = {
    "RED": (255, 0, 0),
    "ORANGE": (255, 165, 0),
    "YELLOW": (255, 255, 0),
    "GREEN": (0, 255, 0),
    "BLUE": (0, 0, 255),
    "BLACK": (0, 0, 0)
}

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )
    
    def forward(self, x):
        return self.network(x)

class BrickBreakerAI:
    def __init__(self, state_size, action_size, learning_rate, discount_rate, batch_size, memory_size):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.batch_size = batch_size
        self.epsilon = 0.1  # Add explicit epsilon value
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = QNetwork(state_size, action_size).to(self.device)
        self.target_model = QNetwork(state_size, action_size).to(self.device)
        self.update_target_model()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=memory_size)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        
        with torch.no_grad():
            self.model.eval()
            action_values = self.model(state_tensor)
            return np.argmax(action_values.cpu().numpy())

    def act(self, state):
        return self.choose_action(state)

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        
        states = torch.FloatTensor(np.array([i[0] for i in minibatch])).to(self.device)
        actions = torch.LongTensor(np.array([i[1] for i in minibatch])).to(self.device)
        rewards = torch.FloatTensor(np.array([i[2] for i in minibatch])).to(self.device)
        next_states = torch.FloatTensor(np.array([i[3] for i in minibatch])).to(self.device)
        dones = torch.FloatTensor(np.array([i[4] for i in minibatch])).to(self.device)

        states = states.squeeze()
        next_states = next_states.squeeze()

        self.model.train()
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
        
        targets = rewards + self.discount_rate * next_q_values * (1 - dones)
        loss = self.criterion(current_q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class BrickBreaker:
    def __init__(self, width=WINDOW_WIDTH, height=WINDOW_HEIGHT, ai_model=None, save_path=None):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Brick Breaker")
        self.clock = pygame.time.Clock()
        self.save_path = save_path or "model.pkl"
        
        self.reset()
        
        self.ai = BrickBreakerAI(STATE_SIZE, ACTION_SIZE, LEARNING_RATE, DISCOUNT_RATE, BATCH_SIZE, MEMORY_SIZE)
        if ai_model is not None:
            self.ai.model.load_state_dict(ai_model)
            self.ai.target_model.load_state_dict(ai_model)

    def create_bricks(self):
        self.brick_list = []
        for row in range(7):
            if row % 3 != 0:
                for col in range(17):
                    if col % 4 != 0:
                        brick = pygame.Rect(
                            20 + col * (BRICK_WIDTH + BRICK_PADDING),
                            50 + row * (BRICK_HEIGHT + BRICK_PADDING),
                            BRICK_WIDTH,
                            BRICK_HEIGHT
                        )
                        self.brick_list.append(brick)

    def reset(self):
        self.score = 0
        self.game_over = False
        self.reward = 0
        
        # Reset ball
        self.ball = pygame.Rect(
            self.width/2 - BALL_SIZE/2,
            self.height - 50,
            BALL_SIZE,
            BALL_SIZE
        )
        self.rnd_dir = random.choice(RND_DIR_LIST)
        self.ball_speed_x = BALL_SPEED * self.rnd_dir
        self.ball_speed_y = -BALL_SPEED
        
        # Reset paddle
        self.paddle = pygame.Rect(
            self.width/2 - PADDLE_WIDTH/2,
            self.height - PADDLE_HEIGHT - 10,
            PADDLE_WIDTH,
            PADDLE_HEIGHT
        )
        
        self.create_bricks()
        self.state = self.get_state()
        self.action = 0
        self.next_state = self.get_state()
        self.done = False

    def save_model(self):
        with open(self.save_path, "wb") as file:
            pickle.dump(self.ai.model.state_dict(), file)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                self.save_model()
                return False
        return True
    
    def get_state(self):
        """
        Returns the current state of the game as a numpy array.
        State consists of: [paddle_x, ball_x, ball_y, ball_speed_x]
        """
        state = np.array([
            self.paddle.x,
            self.ball.x,
            self.ball.y,
            self.ball_speed_x
        ])
        return state.reshape(1, STATE_SIZE)

    def draw_game(self):
        """Draws all game elements to the screen"""
        # Draw bricks
        for brick in self.brick_list:
            pygame.draw.rect(self.screen, COLORS["RED"], brick)
        
        # Draw paddle
        pygame.draw.rect(self.screen, COLORS["YELLOW"], self.paddle)
        
        # Draw ball
        pygame.draw.circle(self.screen, COLORS["GREEN"], 
                        (self.ball.x + BALL_SIZE/2, self.ball.y + BALL_SIZE/2), 
                        BALL_SIZE/2)
        
        # Draw score
        font = pygame.font.SysFont("Arial", 30)
        score_label = font.render(f"Score: {self.score} Reward: {self.reward}", 1, COLORS["ORANGE"])
        self.screen.blit(score_label, (10, 10))

    def move_ball(self):
        """Updates ball position and handles collisions"""
        self.ball.x += self.ball_speed_x * abs(self.rnd_dir)
        self.ball.y += self.ball_speed_y

        # Brick collision check
        index = self.ball.collidelist(self.brick_list)
        if index != -1:
            self.score += REWARD_BRICK_HIT
            self.brick_list.pop(index)
            self.ball_speed_y *= -1

        # Wall collision checks
        if self.ball.x <= 0:
            self.ball.x = 0
            self.ball_speed_x *= -1
        elif self.ball.x >= self.width - BALL_SIZE:
            self.ball.x = self.width - BALL_SIZE
            self.ball_speed_x *= -1

        if self.ball.y <= 0:
            self.ball.y = 0
            self.ball_speed_y *= -1
        elif self.ball.y >= self.height - BALL_SIZE:
            self.game_over = True

        # Paddle collision
        if self.ball.colliderect(self.paddle):
            self.ball.y = self.height - PADDLE_HEIGHT - BALL_SIZE - 10
            self.reward += REWARD_PADDLE_HIT
            self.ball_speed_y *= -1

    def step(self, action):
        """
        Execute one time step within the environment
        
        Args:
            action (int): The action to take (0: move left, 1: move right, 2: stay)
        """
        self.action = action
        if action == 0:  # Move left
            self.paddle.x -= PADDLE_SPEED
        elif action == 1:  # Move right
            self.paddle.x += PADDLE_SPEED
        
        # Keep paddle within screen bounds
        if self.paddle.x <= 0:
            self.paddle.x = 0
        elif self.paddle.x >= self.width - PADDLE_WIDTH:
            self.paddle.x = self.width - PADDLE_WIDTH
    def update(self):
        self.move_ball()
        self.state = self.next_state
        self.next_state = self.get_state()
        
        if len(self.brick_list) == 0:
            self.reward = REWARD_GAME_WIN
            self.done = True
            self.reset()
            
        if self.game_over:
            self.reward = REWARD_GAME_OVER
            self.ai.remember(self.state, self.action, self.reward, self.next_state, self.done)
            self.ai.replay(BATCH_SIZE)
            self.reset()
            
        self.step(self.ai.act(self.state))

    def run(self):
        scores = []
        running = True
        
        while running:
            running = self.handle_events()
            
            self.screen.fill(COLORS["BLACK"])
            self.draw_game()
            self.update()
            
            if self.score >= 144:
                scores.append(self.score)
                self.save_model()
                break
                
            pygame.display.update()
            self.clock.tick(60)
            
        return scores

def main():
    save_path = "v2.pkl"
    ai_model = None
    
    if os.path.exists(save_path):
        print("Loading saved model...")
        with open(save_path, "rb") as file:
            ai_model = pickle.load(file)
            
    game = BrickBreaker(ai_model=ai_model, save_path=save_path)
    scores = game.run()
    print(f"Final scores: {scores}")
    pygame.quit()

if __name__ == "__main__":
    main()