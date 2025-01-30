import gym
import numpy as np
import torch
import random
from collections import deque # double-ended queue for the replay buffer
# import itertools
from torch import nn # nn for building and training neural networks

# Paper
# https://arxiv.org/pdf/1312.5602


# Constants
GAMMA = 0.99  # Discount rate
BATCH_SIZE = 32  # Number of transitions we're going to sample from the replay buffer
BUFFER_SIZE = 50000  # Max number of transitions we will store before overriding old transitions
MIN_REPLAY_SIZE = 1000  # Number of transitions in the replay buffer before we start computing gradients and training
EPSILON_START = 1.0  # Start value of epsilon
EPSILON_END = 0.02  # End value of epsilon
EPSILON_DECAY = 10000  # Decay period
TARGET_UPDATE_FREQ = 1000  # Number of steps where we set the target parameters equal to the online parameters

# Network class (a simple neural network with two layers)
class Network(nn.Module):
    def __init__(self, env):
        super().__init__()

        # Number of neurons in the input layer of the neural network
        # Not needed for cart-pole game (since its 2D)
        in_features = int(np.prod(env.observation_space.shape))

        self.net = nn.Sequential(
            nn.Linear(in_features, 64),  # 64 neurons
            nn.Tanh(),
            nn.Linear(64, env.action_space.n))  # number of possible actions the agent can take, has to be set and known ? 
        
    def forward(self, x): # Another required function
        return self.net(x)
    

    # The act method computes the action to take based on the current observation
    # by forwarding the input and selecting the action with the highest Q-value.
    def act(self, obs):

        # This is how we select an action in Q learning
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        q_values = self(obs_t.unsqueeze(0))  # unsqueeze because there is no batch dimension here
        max_q_index = torch.argmax(q_values, dim=1)[0]  # Fix this line by using correct indexing
        action = max_q_index.detach().item()
        return action # action is just a number




# Create environment and apply new_step_api
env = gym.make('CartPole-v1', new_step_api=True)  # Using the new step API
replay_buffer = deque(maxlen=BUFFER_SIZE) # Stores experiences for training
rew_buffer = deque([0.0], maxlen=100) # Stores rewards
episode_reward = 0.0

# Create online and target networks
# During training, the online network is updated frequently to learn optimal Q-values, while the target
# network is only updated periodically to provide stable target Q-values when computing the loss.

online_net = Network(env) # The online network is the primary network that is actively updated during training. 
target_net = Network(env) # The target network serves as a stable reference for the Q-value targets during training.
target_net.load_state_dict(online_net.state_dict())

# Create optimizer
# Uses the Adam (Adaptive Moment Estimation) optimization algorithm for training
optimizer = torch.optim.Adam(online_net.parameters(), lr=5e-4)

# Initialize replay buffer
obs = env.reset()
for _ in range(MIN_REPLAY_SIZE):
    action = env.action_space.sample()

    # Edited
    new_obs, rew, terminated, truncated, _ = env.step(action)
    done = terminated or truncated  # Combine to create a `done` flag
    transition = (obs, action, rew, done, new_obs)
    replay_buffer.append(transition)  # Append to the replay buffer
    obs = new_obs

    if done:
        obs = env.reset()



#✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆
#✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆✯☆


# Set a maximum number of steps for training
MAX_STEPS = 29000

# Training loop
obs = env.reset()

for step in range(MAX_STEPS):
    # Epsilon greedy
    epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
    rnd_sample = random.random()

    if rnd_sample <= epsilon:
        action = env.action_space.sample()  # Explore (random action)
    else:
        action = online_net.act(obs)  # Exploit (choose action based on the network)

    # print(f"Step {step}: Action taken = {action}") # Action is a number betwee 0 and 1 (righ or left)

    # Edited
    new_obs, rew, terminated, truncated, _ = env.step(action) # updates the environment based on that action?
    done = terminated or truncated  # Combine to create a `done` flag
    transition = (obs, action, rew, done, new_obs)  # Transition tuple
    replay_buffer.append(transition)  # Append to the replay buffer
    obs = new_obs

    episode_reward += rew # Reward is Accumulated during the episode and stored in rew_buffer when the episode ends

    if done:
        obs = env.reset()
        rew_buffer.append(episode_reward)
        episode_reward = 0.0

        # After we solve the problem, render it to watch it play
        if len(rew_buffer) >= 100: # This line makes sure theres at leat 100 episodes' worth of rewards inside rew_buffer
            if np.mean(rew_buffer) >= 195: # This line check to see if average of all those 100 eposodes' rewards is at least 195, indicating that 
                                           # the agent has "solved" the problem
                while True: 
                    action = online_net.act(obs) # Key line, continuously plays the game using the online_net's
                                                 # learned policy and renders each step until the environment signals termination
                    obs, _, terminated, truncated, _ = env.step(action) # Edited
                    done = terminated or truncated
                    env.render()
                    if done:
                        env.reset()
                        break  # Break out of the while loop after reset

    # Start Gradient Step
    if len(replay_buffer) >= BATCH_SIZE: # Edited
        transitions = random.sample(replay_buffer, BATCH_SIZE) # Randomply select a batch of transitions from
                                                               # the repay buffer

        # Breaks the batch of transitions into separate arrays for states, actions, rewards, done flags, and next states
        obses = np.array([t[0] for t in transitions])
        actions = np.array([t[1] for t in transitions])
        rews = np.array([t[2] for t in transitions])
        dones = np.array([t[3] for t in transitions])
        new_obses = np.array([t[4] for t in transitions])

        # Convert them to torch tensors for easier calculation (I think)
        # Neural networks in PyTorch operate on tensors
        obses_t = torch.as_tensor(obses, dtype=torch.float32)
        actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)
        rews_t = torch.as_tensor(rews, dtype=torch.float32).unsqueeze(-1)
        dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1)
        new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32)

        # Compute targets for the loss function
        target_q_values = target_net(new_obses_t)
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

        targets = rews_t + GAMMA * (1 - dones_t) * max_target_q_values # the Bellman equation

        # Compute loss
        q_values = online_net(obses_t)
        action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)

        # Edited
        # Huber loss is computed between the predicted Q-values (action_q_values)
        # and the target Q-values (targets)
        loss = nn.SmoothL1Loss()(action_q_values, targets)

        # Gradient Descent ( No idea)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Update Target Network
    # The target network is updated periodically to maintain stability?
    if step % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(online_net.state_dict())

    # Logging to monitor progress
    if step % 1000 == 0:
        print(f"Step {step}")
        print(f"Average Reward: {np.mean(rew_buffer)}")

    # Stop training at step 29000
    if step >= 29000:
        print("Training stopped at step 29000.")
        break