from torch import nn
import torch 
import gym
from collections import deque #??
import itertools
import numpy as np
import random


# Unedited code

GAMMA= 0.99 # Discout rate
BATCH_SIZE=32 # Number of transitions were going to sample from the repay buffer
BUFFER_SIZE=50000 # Max number of transitions we will store before overriding old transitions 
MIN_REPLAY_SIZE=1000 # number of transitions in the repay buffer before we start computing gradiants and training 
EPSILON_START=1.0 # Start val of epsilon
EPSILON_END=0.02 # End value of epsilon
EPSILON_DECAY=10000 # Decay period (how epsilon will linearly decay from EPSILON_START to EPSILON_END)
TARGET_UPDATE_FREQ=1000 # Number of steps where we set the target parameters equal to the online parameters

# Network class
class Network(nn.Module):
    def __init__(self,env):
        super().__init__()

        # Number of neurons in the input layer of the neural network
        # Not needed for cart-pole game (since its 2D)
        in_features = int(np.prod(env.observation_space.shape))

        self.net = nn.Sequential(
            nn.Linear(in_features, 64), #64 layers
            nn.Tanh(),
            nn.Linear(64, env.action_space.n)) # number of possible actions that the agant can take
        
    def forward (self, x): # Another required function
        return self.net(x)
    
    def act (self, obs) :

        # This is how we select an action in Q learning
        obs_t = torch.as_tensor(obs, dtype=torch.float32)    
        q_values = self(obs_t.unsqueeze(0)) # unsqueeze bc there is no batch dimention here

        max_q_index = torch.argmax(q_values, dim=1 )[0]
        action = max_q_index.detach().item() 

        return action # Just a number 

# Environment using gym
env = gym.make('CartPole-v0')
# Keep track of...
replay_buffer = deque(maxlen=BUFFER_SIZE)
rew_buffer = deque([0.0], maxlen=100)
episode_reward = 0.0

# Create online network
online_net = Network(env) 
# Create target network
target_net = Network(env)

# Set the target network parameters to the online network parameters (a part of the algorithm)
target_net.load_state_dict(online_net.state_dict())

# Create optimizer
optimizer = torch.optim.Adam(online_net.parameters(), lr=5e-4)

# Initialize replay buffer
obs = env.reset()
for _ in range(MIN_REPLAY_SIZE):
    action = env.action_space.sample()

    new_obs, rew, done, _ = env.step(action) # Gives an error, older version?
    #new_obs, rew, terminated, truncated, info = env.step(action)
    #done = terminated or truncated

    transition = (obs, action, rew, done, new_obs) # Transition tuple
    replay_buffer.append(transition) # Append it to the replay buffer
    obs = new_obs

    if done:
        obs = env.reset()


# Training loop
obs = env.reset()

for step in itertools.count():
    epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END]) # Numpy function, prety straightforward, it will start at epsilon start and end at epsilon end after epsilon decay steps
    # As more episodes go by, we do less exploring and more exploiting

rnd_sample = random.random()


if rnd_sample <= epsilon:
    action = env.action_space.sample()
# if not, sleect an action based on the networks
else:
    action = online_net.act(obs)

new_obs, rew, done, _ = env.step(action)
transition = (obs, action, rew, done, new_obs) # Transition tuple
replay_buffer.append(transition) # Append it to the replay buffer
obs = new_obs

episode_reward +=rew

if done:
    obs = env.reset()

    rew_buffer.append(episode_reward)
    episode_reward = 0.0

# Missing code here


# Start Gradiant Step
#missign code

# Select batch size number of random transitions from our replay buffer
transitions = random.sample(replay_buffer, BATCH_SIZE)

obses = np.asanyarray([t[0] for t in transitions])
actions = np.asanyarray([t[1] for t in transitions])
rews = np.asanyarray([t[2] for t in transitions])
dones = np.asanyarray([t[3] for t in transitions])
new_obses = np.asanyarray([t[4] for t in transitions])

# Pytorch tensors
obses_t = torch.as_tensor(obses, dtype=torch.float32)
actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)
rews_t = torch.as_tensor(rews, dtype=torch.float32).unsqueeze(-1)
dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1)
new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32)

# Compute targets for our loss functions

target_q_values = target_net(new_obses_t)
# Here we have a set of Q values for EACH observation
# Q values are dimention 1, get the max value  in dimention one and discard the rest, and keep that dimention oen even though there is only one value in it
max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

# Piece-wise function on thr paper
targets = rews_t + GAMMA *(1- dones_t) * max_target_q_values

# Compute loss

#  Set of Q values for each observation
q_values = online_net(obses_t)

# Gives us the predicted q value for the action we took
action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)

loss = nn.smooth_ll_loss(action_q_values, targets)

# Gradient Descent
optimizer.zero_grad()
loss.backward()
optimizer.step()

# Update Target Network
if step % TARGET_UPDATE_FREQ ==0:
    target_net.load_state_dict(online_net.state_dict())

# Logging to see if improvement is happening
if step %1000 == 0:
    print()
    print('Step', step)
    print('average Reward', np.mean(rew_buffer))