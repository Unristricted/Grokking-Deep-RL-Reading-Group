import pygame
import random
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


# Parameters
STATE_SIZE = 4
ACTION_SIZE = 3
LEARNING_RATE = 0.001
DISCOUNT_RATE = 0.99
BATCH_SIZE = 32
MEMORY_SIZE = 1000000
BALL_SPEED = 10
RND_DIR_LIST = ( -1.1, -1.0, -0.9, -0.8, -0.7, -0.6,-0.5, -0.4, -0.3, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1)


# Colors
colors = {
    "red": (255, 0, 0),
    "orange": (255, 165, 0),
    "yellow": (255, 255, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "black": (0, 0, 0)
}

# Neural Network Model
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

# AI Class
class BrickBreakerAI:
    def __init__(self, state_size, action_size, learning_rate, discount_rate, batch_size, memory_size):
        """
        Initialize the Deep Q-Learning Agent
        
        Args:
        - state_size: Dimensionality of game state representation
        - action_size: Number of possible actions the agent can take
        - learning_rate: Speed at which the agent updates its knowledge
        - discount_rate: Importance of future rewards vs immediate rewards
        - batch_size: Number of experiences processed in each learning iteration
        - memory_size: Total historical experiences the agent can remember
        """
        # Core learning configuration
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.batch_size = batch_size
        
        # Select most efficient computational device (GPU/CPU)
        # This allows for faster neural network computations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Neural Network Architecture Setup
        # Two networks prevent learning instability:
        # 1. self.model: Primary network that actively learns
        # 2. self.target_model: Stable reference network for consistent evaluation
        self.model = QNetwork(state_size, action_size).to(self.device)
        self.target_model = QNetwork(state_size, action_size).to(self.device)
        self.update_target_model()  # Initial synchronization of networks
        
        # Optimizer: Adjusts network weights to minimize prediction error
        # Adam dynamically adapts learning rates for each parameter
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Loss Function: Measures difference between predicted and actual rewards
        # Mean Squared Error helps quantify learning progress
        self.criterion = nn.MSELoss()
        
        # Experience Memory: Stores game interactions for later learning
        # Using deque ensures memory doesn't grow indefinitely
        self.memory = deque(maxlen=memory_size)

    def update_target_model(self):
        """
        Synchronizes target network with main network
        
        Conceptual Purpose:
        - Provides a stable reference point for learning
        - Prevents rapid, potentially destabilizing updates
        - Creates a 'snapshot' of current learning state
        """
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """
        Stores individual game experiences for later learning
        
        Each memory tuple captures a moment of game interaction:
        - state: Current game state before action
        - action: Selected paddle movement
        - reward: Immediate feedback from the action
        - next_state: Game state after taking the action
        - done: Whether this interaction ended the game episode
        """
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state, epsilon):
        """
        Implements Epsilon-Greedy Action Selection Strategy
        
        Args:
        - state: Current game state
        - epsilon: Probability of choosing a random action
        
        Returns:
        - Selected action (paddle movement)
        """
        # Convert state to tensor for neural network processing
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        # Exploration Phase: Randomly select action
        if np.random.rand() <= epsilon:
            return np.random.randint(self.action_size)
        
        # Exploitation Phase: Use neural network to predict best action
        with torch.no_grad():  # Prevents unnecessary gradient computation
            self.model.eval()  # Set model to evaluation mode
            action_values = self.model(state_tensor)
            
            # Select action with highest predicted reward
            return np.argmax(action_values.cpu().numpy())

    def act(self, state):
        """
        Wrapper method for action selection with default exploration rate
        Provides a simple interface for taking actions during gameplay
        """
        return self.choose_action(state, epsilon=0.1)

    def replay(self, batch_size):
        """
        Core Learning Method: Experience Replay
        
        Conceptual Learning Process:
        1. Sample random past experiences
        2. Compute predicted vs actual rewards
        3. Update neural network to improve future predictions
        
        Prevents overfitting and provides stable learning trajectory
        """
        # Only learn if enough experiences are available
        if len(self.memory) < self.batch_size:
            return
        
        # Randomly sample experiences to break correlation
        minibatch = random.sample(self.memory, self.batch_size)
        
        # Prepare batch data as tensors for neural network processing
        states = torch.FloatTensor(np.array([i[0] for i in minibatch])).to(self.device)
        actions = torch.LongTensor(np.array([i[1] for i in minibatch])).to(self.device)
        rewards = torch.FloatTensor(np.array([i[2] for i in minibatch])).to(self.device)
        next_states = torch.FloatTensor(np.array([i[3] for i in minibatch])).to(self.device)
        dones = torch.FloatTensor(np.array([i[4] for i in minibatch])).to(self.device)
        
        # Ensure tensor dimensions are correct
        states = states.squeeze()
        next_states = next_states.squeeze()
        
        # Predict current state's Q-values
        self.model.train()
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Compute stable target Q-values using target network
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
        
        # Calculate target values considering future rewards and episode termination
        targets = rewards + self.discount_rate * next_q_values * (1 - dones)
        
        # Compute loss: difference between predicted and target Q-values
        loss = self.criterion(current_q_values, targets)
        
        # Perform learning update
        self.optimizer.zero_grad()  # Clear previous gradients
        loss.backward()  # Compute gradient of loss
        self.optimizer.step()  # Update network weights

class BrickBreaker:
    def __init__(self, width, height, ai_model=None):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Brick Breaker")
        pygame.font.init()
        self.clock = pygame.time.Clock()
        self.score = 0
        self.game_over = False
        
        # Ball setup
        self.ball = pygame.Rect(self.width/2-10, self.height-50, 20, 20)
        self.rnd_dir = random.choice(RND_DIR_LIST)
        self.ball_speed_x = BALL_SPEED * self.rnd_dir
        self.ball_speed_y = -BALL_SPEED
        
        # Paddle setup
        self.paddle = pygame.Rect(self.width/2-70, self.height-20, 140, 10)
        
        # Bricks setup
        self.brick_list = []
        self.create_bricks()

        # AI Initialization
        self.ai = BrickBreakerAI(STATE_SIZE, ACTION_SIZE, LEARNING_RATE, DISCOUNT_RATE, BATCH_SIZE, MEMORY_SIZE)
        
        # Load pre-trained model if provided
        if ai_model is not None:
            self.ai.model.load_state_dict(ai_model)
            self.ai.target_model.load_state_dict(ai_model)
        
        self.state = self.get_state()
        self.action = 0
        self.reward = 0
        self.next_state = self.get_state()
        self.done = False

    def create_bricks(self):
        for row in range(7):
            if row % 3 != 0:
                for col in range(17):
                    if col % 4 != 0:
                        brick = pygame.Rect(20 + col * (40 + 5), 50 + row * (20 + 5), 40, 20)
                        self.brick_list.append(brick)

    def draw_bricks(self):
        for brick in self.brick_list:
            pygame.draw.rect(self.screen, colors["red"], brick)

    def draw_paddle(self):
        pygame.draw.rect(self.screen, colors["yellow"], self.paddle)

    def draw_ball(self):
        pygame.draw.circle(self.screen, colors["green"], (self.ball.x + 10, self.ball.y + 10), 10)

    def move_ball(self):
        self.ball.x += self.ball_speed_x * abs(self.rnd_dir)
        self.ball.y += self.ball_speed_y

        # Brick collision check
        index = self.ball.collidelist(self.brick_list)
        if index != -1:
            self.score += 3
            self.brick_list.pop(index)
            self.ball_speed_x *= -1

        # Wall collision checks
        if self.ball.x <= 0:
            self.ball.x = 0
            self.ball_speed_x *= -1
        elif self.ball.x >= self.width - 20:
            self.ball.x = self.width - 20
            self.ball_speed_x *= -1

        if self.ball.y <= 0:
            self.ball.y = 0
            self.ball_speed_y *= -1
        elif self.ball.y >= self.height - 20:
            self.ball.y = self.height - 20
            self.ball_speed_y *= -1
            self.game_over = True

        # Paddle collision
        if self.ball.colliderect(self.paddle):
            self.ball.y = self.height - 50
            self.reward += 3
            self.ball_speed_y *= -1
            if self.ball_speed_x > 0:
                self.ball_speed_x = BALL_SPEED * abs(random.choice(RND_DIR_LIST))
            else:
                self.ball_speed_x = BALL_SPEED * -abs(random.choice(RND_DIR_LIST))
    def reset(self):
        self.score = 0
        self.game_over = False
        self.ball.x = self.width/2-10
        self.ball.y = self.height-50
        self.rnd_dir = random.choice(RND_DIR_LIST)
        self.ball_speed_x = BALL_SPEED * self.rnd_dir
        self.ball_speed_y = -BALL_SPEED
        self.paddle.x = self.width/2-70
        self.brick_list = []
        self.create_bricks()

    def get_state(self):
        state = np.array([self.paddle.x, self.ball.x, self.ball.y, self.ball_speed_x])
        state = state.reshape(1, STATE_SIZE)
        return state

    def step(self, action):
        middle = self.width/2-70
        self.action = action
        if self.action == 0:
            self.paddle.x -= 10
        elif self.action == 1:
            self.paddle.x += 10
    
        # Keep paddle within screen bounds
        if self.paddle.x <= 0:
            self.paddle.x = 0
            self.reward -= 0.25
        elif self.paddle.x >= self.width - 140:
            self.paddle.x = self.width - 140
            self.reward -= 0.25

        # Reward Being in Middle and tracking ball
        if self.paddle.x >= middle - 10 and self.paddle.x <= middle + 10:
            self.reward += 0.01

        if self.ball.x + 10 >= self.paddle.x+10 and self.ball.x <= self.paddle.x + 130:
            self.reward += 0.01
        

    def run(self):
        tempArray = []
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if (event.type==pygame.KEYDOWN and event.key ==pygame.K_s):
                    # Save the model
                    with open("C:/Users/jj720/Lab/nohope/modelbigbig.pkl", "wb") as file:
                        pickle.dump(self.ai.model.state_dict(), file)
                    print(tempArray)
                    pygame.quit()
                    sys.exit()

            
            self.screen.fill(colors["black"])
            self.draw_bricks()
            self.draw_paddle()
            self.draw_ball()
            self.move_ball()
            
            self.state = self.next_state
            self.next_state = self.get_state()
            self.done = False

            if len(self.brick_list) == 0:
                self.reset()
                self.done = True
                self.reward += 100
            
            if self.game_over:
                tempArray.append(self.score)
                self.reward = -100
                self.ai.remember(self.state, self.action, self.reward, self.next_state, self.done)
                self.ai.replay(BATCH_SIZE)
                self.reward = 0
                self.reset()
                self.done = True
                
            # Draw score
            font = pygame.font.SysFont("Arial", 30)
            score_label = font.render(f"Score: {self.score} Reward:{round(self.reward, 1)}", 1, colors["orange"])
            self.screen.blit(score_label, (10, 10))
            
            self.step(self.ai.act(self.state))
            pygame.display.update()
            self.clock.tick(60)

            if event.type==pygame.KEYDOWN and event.key ==pygame.K_s:
                # Save the model
                print(tempArray)
                with open("C:/Users/jj720/Lab/nohope/modelbigbig.pkl", "wb") as file:
                    pickle.dump(self.ai.model.state_dict(), file)
                sys.exit()

# Main function
def main():
    import os 
    if os.path.exists("C:/Users/jj720/Lab/nohope/modelbigbig.pkl"):
        print("Using Saved")
        with open("C:/Users/jj720/Lab/nohope/modelbigbig.pkl", "rb") as file:
            ai_model = pickle.load(file)
        game = BrickBreaker(800, 600, ai_model=ai_model)
    else:
        game = BrickBreaker(800, 600)
    game.run()

if __name__ == "__main__":
    main()