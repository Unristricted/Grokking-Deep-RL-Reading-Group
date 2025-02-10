import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# Hyperparameters
EPISODES = 2000
GAMMA = 0.99
LR = 0.002
EPS_CLIP = 0.2
K_EPOCHS = 4
UPDATE_INTERVAL = 200

# Output: A probability distribution over all actions.
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
# Required function
    def forward(self, x):
        return self.fc(x)

# Similar to the policy network but outputs a single value (scalar).
class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.fc(x)


# policy (updated during training) and policy_old (baseline used for PPO update).
class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.policy = PolicyNetwork(state_dim, action_dim).to(device)
        self.policy_old = PolicyNetwork(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=LR)

        self.value = ValueNetwork(state_dim).to(device)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=LR)

        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.done_flags = []


    def select_action(self, state):
        state = torch.FloatTensor(state).to(device)
        with torch.no_grad():
            probabilities = self.policy_old(state)
            dist = Categorical(probabilities)
            action = dist.sample()

        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(dist.log_prob(action))

        return action.item()


# This stores rewards and terminal flags for each timestep.
    def store_reward(self, reward, done):
        self.rewards.append(reward)
        self.done_flags.append(done)

# Normalizes returns to stabilize training.
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


# Computes normalized returns.
    def update(self):
        states = torch.stack(self.states).detach().to(device)
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




# Main Training Loop
env = gym.make('CartPole-v1')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = PPOAgent(state_dim, action_dim)

episode_rewards = []

for episode in range(EPISODES):
    state = env.reset()
    total_reward = 0

    for t in range(1, 10000):
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)

        agent.store_reward(reward, done)
        state = next_state
        total_reward += reward

        if done:
            break

    episode_rewards.append(total_reward)

    if episode % UPDATE_INTERVAL == 0:
        agent.update()

    if (episode + 1) % 100 == 0:
        avg_reward = np.mean(episode_rewards[-100:])
        print(f"Episode {episode + 1}, Average Reward: {avg_reward:.2f}")

print("Training completed")
env.close()
