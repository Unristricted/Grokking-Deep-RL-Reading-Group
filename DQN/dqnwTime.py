import gym
import numpy as np
import torch
import random
from collections import deque
import time  # Import the time module
from torch import nn

# Constants
GAMMA = 0.99
BATCH_SIZE = 32
BUFFER_SIZE = 50000
MIN_REPLAY_SIZE = 1000
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 10000
TARGET_UPDATE_FREQ = 1000

class Network(nn.Module):
    def __init__(self, env):
        super().__init__()
        in_features = int(np.prod(env.observation_space.shape))
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.Tanh(),
            nn.Linear(64, env.action_space.n))
        
    def forward(self, x):
        return self.net(x)
    
    def act(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        q_values = self(obs_t.unsqueeze(0))
        max_q_index = torch.argmax(q_values, dim=1)[0]
        action = max_q_index.detach().item()
        return action

# Create environment
env = gym.make('CartPole-v1', new_step_api=True)
replay_buffer = deque(maxlen=BUFFER_SIZE)
rew_buffer = deque([0.0], maxlen=100)
episode_reward = 0.0

# Create online and target networks
online_net = Network(env)
target_net = Network(env)
target_net.load_state_dict(online_net.state_dict())

# Create optimizer
optimizer = torch.optim.Adam(online_net.parameters(), lr=5e-4)

# Initialize replay buffer
obs = env.reset()
for _ in range(MIN_REPLAY_SIZE):
    action = env.action_space.sample()
    new_obs, rew, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    transition = (obs, action, rew, done, new_obs)
    replay_buffer.append(transition)
    obs = new_obs
    if done:
        obs = env.reset()

# Training loop
MAX_STEPS = 29000
obs = env.reset()
highest_time = 0  # Track highest time
episode_start_time = time.time()  # Start time for the current episode

for step in range(MAX_STEPS):
    epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
    rnd_sample = random.random()

    if rnd_sample <= epsilon:
        action = env.action_space.sample()
    else:
        action = online_net.act(obs)

    new_obs, rew, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    transition = (obs, action, rew, done, new_obs)
    replay_buffer.append(transition)
    obs = new_obs

    episode_reward += rew

    if done:
        episode_time = time.time() - episode_start_time  # Calculate episode duration
        print(f"Episode finished. Time kept pole up: {episode_time:.2f} seconds")
        
        if episode_time > highest_time:
            highest_time = episode_time  # Update highest time if needed
            print(f"New highest time: {highest_time:.2f} seconds!")
        else:
            print(f"Highest time remains: {highest_time:.2f} seconds")
        
        obs = env.reset()
        rew_buffer.append(episode_reward)
        episode_reward = 0.0
        episode_start_time = time.time()  # Reset the start time for the new episode

        if len(rew_buffer) >= 100 and np.mean(rew_buffer) >= 195:
            while True:
                action = online_net.act(obs)
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                env.render()
                if done:
                    env.reset()
                    break

    if len(replay_buffer) >= BATCH_SIZE:
        transitions = random.sample(replay_buffer, BATCH_SIZE)
        obses = np.array([t[0] for t in transitions])
        actions = np.array([t[1] for t in transitions])
        rews = np.array([t[2] for t in transitions])
        dones = np.array([t[3] for t in transitions])
        new_obses = np.array([t[4] for t in transitions])

        obses_t = torch.as_tensor(obses, dtype=torch.float32)
        actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)
        rews_t = torch.as_tensor(rews, dtype=torch.float32).unsqueeze(-1)
        dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1)
        new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32)

        target_q_values = target_net(new_obses_t)
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
        targets = rews_t + GAMMA * (1 - dones_t) * max_target_q_values

        q_values = online_net(obses_t)
        action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)
        loss = nn.SmoothL1Loss()(action_q_values, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if step % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(online_net.state_dict())

    if step % 1000 == 0:
        print(f"Step {step}")
        print(f"Average Reward: {np.mean(rew_buffer)}")

    if step >= 29000:
        print("Training stopped at step 29000.")
        break
