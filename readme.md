# Reinforcement Learning Reading Group

This repository contains implementations and experiments from our Reinforcement Learning reading group, covering fundamental RL algorithms and concepts through practical implementations. The references used were as follows:
- "Reinforcement Learning: An Introduction" by Richard S. Sutton and Andrew G. Barto
- "Grokking Deep Reinforcement Learning" by Miguel Morales
- "Algorithms for Reinforcement Learning" by Csaba Szepesvari
- "Playing Atari with Deep Reinforcement Learning" by Volodymyr Mnih and friends
- "Addressing Function Approximation Error in Actor-Critic Methods" by Fujimoto and friends
- "Proximal Policy Optimization Algorithms" by John Schulman and friends
- "Numerical Optimization" by Jorge Nocedal and Stephen J. Wright
- "Probalistic Machine Learning: An Introduction" by Kevin Murphy

When we were wrapping up the reading group, a great reference appeared on Arxiv:
- "Reinforcement Learning: An Overview" by Kevin Murphy

## Repository Structure

The repository is organized into four main sections, each focusing on different aspects of reinforcement learning:

### DP-MDP (Dynamic Programming Based Algorithms)
Comprehensive collection of algorithms when we assume the underlying Markov-Decision-Process in known. 

### DQN (Deep Q-Networks)
Implementation of DQN applied to the Brick Breaker game environment. Features both TensorFlow and PyTorch implementations with various improvements and refinements.

### PPO (Proximal Policy Optimization)
Implementation of PPO demonstrated on the CartPole environment from OpenAI Gym.

### TorchRL
Educational notebooks exploring TorchRL's capabilities and implementation details.


## Getting Started

Each folder contains its own detailed README with implementation details. The implementations progress from fundamental concepts (MDPs) to more advanced algorithms (PPO).

## Prerequisites

- PyTorch
- TensorFlow (for some DQN implementations)
- OpenAI Gym
- TorchRL

## License

MIT License

## Acknowledgments

Special thanks to the following contributors for their work preparing and presenting the reading group materials:

- Alperen Ergur
- Jonathan de Koning
- Melika Golestani
- Vinny Miller
- Thanuka Wijenayaka
- Yaseen Syed
