# DQN Implementation for Brick Breaker

This repository contains an implementation of Deep Q-Learning (DQN) applied to the classic Brick Breaker game. The project was developed as part of a reading group focused on reinforcement learning.

### Files

- `main.py` - Initial implementation using TensorFlow/Keras for the DQN agent. Contains both the game logic and the neural network implementation.

- `maintorch.py` - PyTorch implementation of the DQN agent with improved reward mechanisms and state handling. Includes detailed comments explaining the learning process and neural network architecture.

- `v2.py` - Refined implementation with better code organization, consistent naming conventions, and additional configuration options. Features improved constants management and cleaner class structures.

- `plts.py` - Plotting utility script that generates visualizations of the
agent's performance.

The DQN implementation includes:
- State space of 4 dimensions (paddle position, ball position x/y, ball speed)
- Action space of 3 dimensions (move left, move right, stay)
- Experience replay buffer for stable learning
- Target network for stable Q-value estimation
- Customizable reward structure

The game environment provides:
- Configurable window size and game parameters
- Real-time visualization of gameplay
- Score and reward tracking
- Collision detection and physics
