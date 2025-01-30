import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import cv2
from collections import deque

# Hyperparameters
EPISODES = 1000
GAMMA = 0.99
LR = 0.0003
EPS_CLIP = 0.2
K_EPOCHS = 4
UPDATE_INTERVAL = 1024
STACK_SIZE = 4

class Preprocessor:
    def __init__(self):
        self.frames = deque(maxlen=STACK_SIZE)

    def process_frame(self, frame):
        # Convert to grayscale and resize
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
        return frame / 255.0

    def stack_frames(self, frame, reset=False):
        if reset or len(self.frames) == 0:
            self.frames.extend([frame] * STACK_SIZE)
        else:
            self.frames.append(frame)
        return np.stack(self.frames, axis=0)

class PolicyNetwork(nn.Module):
    def __init__(self, action_dim):
        super(PolicyNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(STACK_SIZE, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class ValueNetwork(nn.Module):
    def __init__(self):
        super(ValueNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(STACK_SIZE, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class PPOAgent:
    def __init__(self, action_dim):
        self.policy = PolicyNetwork(action_dim).to(device)
        self.policy_old = PolicyNetwork(action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=LR)

        self.value = ValueNetwork().to(device)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=LR)

        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.done_flags = []

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            probabilities = self.policy_old(state)
            dist = Categorical(probabilities)
            action = dist.sample()

        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(dist.log_prob(action))

        return action.item()

    def store_reward(self, reward, done):
        self.rewards.append(reward)
        self.done_flags.append(done)

    def compute_returns(self):
        returns = []
        discounted_sum = 0
        for reward, done in zip(reversed(self.rewards), reversed(self.done_flags)):
            if done:
                discounted_sum = 0
            discounted_sum = reward + GAMMA * discounted_sum
            returns.insert(0, discounted_sum)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        return returns

    def update(self):
        states = torch.cat(self.states).detach().to(device)
        actions = torch.stack(self.actions).detach().to(device)
        old_log_probs = torch.stack(self.log_probs).detach().to(device)
        returns = self.compute_returns()

        for _ in range(K_EPOCHS):
            log_probs, state_values = self.evaluate(states, actions)
            ratios = torch.exp(log_probs - old_log_probs.detach())

            advantages = returns - state_values.detach().squeeze()

            surrogate1 = ratios * advantages
            surrogate2 = torch.clamp(ratios, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantages
            policy_loss = -torch.min(surrogate1, surrogate2).mean()

            value_loss = nn.MSELoss()(state_values.squeeze(), returns)

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.clear_memory()

    def evaluate(self, states, actions):
        probabilities = self.policy(states)
        dist = Categorical(probabilities)
        log_probs = dist.log_prob(actions)

        state_values = self.value(states)

        return log_probs, state_values

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.done_flags = []

# Main Loop
env = gym.make("Breakout-v4")  # Use gymnasium's environment directly

# Wrap the environment with Atari preprocessing
env = gym.wrappers.AtariPreprocessing(env, frame_skip=4, grayscale_newaxis=True, scale_obs=True)

preprocessor = Preprocessor()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

action_dim = env.action_space.n
agent = PPOAgent(action_dim)

episode_rewards = []

for episode in range(EPISODES):
    state, _ = env.reset()  # gymnasium reset returns state and info, using _ to ignore info
    state = preprocessor.process_frame(state)
    state = preprocessor.stack_frames(state, reset=True)
    total_reward = 0

    for t in range(1, 10000):
        action = agent.select_action(state)
        next_state, reward, done, _, _ = env.step(action)  # gymnasium returns 5 values
        next_state = preprocessor.process_frame(next_state)
        next_state = preprocessor.stack_frames(next_state)

        agent.store_reward(reward, done)
        state = next_state
        total_reward += reward

        if done:
            break

    episode_rewards.append(total_reward)

    if episode % UPDATE_INTERVAL == 0:
        agent.update()

    if (episode + 1) % 10 == 0:
        avg_reward = np.mean(episode_rewards[-10:])
        print(f"Episode {episode + 1}, Average Reward: {avg_reward:.2f}")

print("Training complete!")
env.close()

