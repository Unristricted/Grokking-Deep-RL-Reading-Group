# PPO Implementation

This repository contains an implementation of Proximal Policy Optimization (PPO), a reinforcement learning algorithm, created for our reading group. The implementation demonstrates PPO on the CartPole environment from OpenAI Gym.

## PPO.py
The main implementation file containing:
- `PolicyNetwork`: Neural network that learns the agent's policy, outputting action probabilities
- `ValueNetwork`: Neural network that estimates state values
- `PPOAgent`: Core PPO implementation including:
  - Action selection
  - Experience storage
  - PPO update logic with clipped objective
  - Return computation and normalization

Key hyperparameters:
- Episodes: 2000
- Gamma (discount factor): 0.99
- Learning rate: 0.002
- Epsilon clip: 0.2
- K epochs: 4
- Update interval: 200

The implementation uses PyTorch and includes standard PPO features like:
- Separate policy and value networks
- Policy clipping
- Advantage normalization
- Value function learning
- Experience buffer management

## Usage

The code is set up to train on the CartPole-v1 environment. The training loop:
1. Collects experience using the current policy
2. Updates the policy every 200 episodes
3. Prints average reward every 100 episodes
