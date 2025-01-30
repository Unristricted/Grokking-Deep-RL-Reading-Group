import gym
import numpy as np
import torch
import random
from collections import deque
import itertools
from torch import nn

# Constants
GAMMA = 0.99  # Discount rate
BATCH_SIZE = 32  # Number of transitions we're going to sample from the replay buffer
BUFFER_SIZE = 50000  # Max number of transitions we will store before overriding old transitions
MIN_REPLAY_SIZE = 1000  # Number of transitions in the replay buffer before we start computing gradients and training
EPSILON_START = 1.0  # Start value of epsilon
EPSILON_END = 0.02  # End value of epsilon
EPSILON_DECAY = 10000  # Decay period
TARGET_UPDATE_FREQ = 1000  # Number of steps where we set the target parameters equal to the online parameters

# Network class
class Network(nn.Module):
    def __init__(self, env):
        super().__init__()

        in_features = int(np.prod(env.observation_space.shape))

        self.net = nn.Sequential(
            nn.Linear(in_features, 64),  # 64 layers
            nn.Tanh(),
            nn.Linear(64, env.action_space.n))  # number of possible actions the agent can take
        
    def forward(self, x):
        return self.net(x)
    
    def act(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        q_values = self(obs_t.unsqueeze(0))  # unsqueeze because there is no batch dimension here
        max_q_index = torch.argmax(q_values, dim=1)[0]  # Fix this line by using correct indexing
        action = max_q_index.detach().item()
        return action

# Create environment and apply new_step_api
env = gym.make('CartPole-v1', new_step_api=True)  # Using the new step API
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
    done = terminated or truncated  # Combine to create a `done` flag
    transition = (obs, action, rew, done, new_obs)
    replay_buffer.append(transition)  # Append to the replay buffer
    obs = new_obs

    if done:
        obs = env.reset()

# Training loop
obs = env.reset()

for step in itertools.count():
    epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
    rnd_sample = random.random()

    if rnd_sample <= epsilon:
        action = env.action_space.sample()  # Explore (random action)
    else:
        action = online_net.act(obs)  # Exploit (choose action based on the network)

    new_obs, rew, terminated, truncated, _ = env.step(action)
    done = terminated or truncated  # Combine to create a `done` flag
    transition = (obs, action, rew, done, new_obs)  # Transition tuple
    replay_buffer.append(transition)  # Append to the replay buffer
    obs = new_obs

    episode_reward += rew

    if done:
        obs = env.reset()
        rew_buffer.append(episode_reward)
        episode_reward = 0.0

        # After we solve the problem, render it to watch it play
        if len(rew_buffer) >= 100:
            if np.mean(rew_buffer) >= 195:
                while True:
                    action = online_net.act(obs)
                    obs, _, done,_= env.step (action)
                    env.render ( )
                    if done:
                        env.reset ()

    # Start Gradient Step
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

        # Compute targets for the loss function
        target_q_values = target_net(new_obses_t)
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

        targets = rews_t + GAMMA * (1 - dones_t) * max_target_q_values

        # Compute loss
        q_values = online_net(obses_t)
        action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)

        loss = nn.SmoothL1Loss()(action_q_values, targets)

        # Gradient Descent
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Update Target Network
    if step % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(online_net.state_dict())

    # Logging to monitor progress
    if step % 1000 == 0:
        print(f"Step {step}")
        print(f"Average Reward: {np.mean(rew_buffer)}")
