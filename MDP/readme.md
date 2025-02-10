# MDP Reading Group Code

This repository contains code implementations and experiments for different MDP algorihtms and environments. The code explores various concepts including multi-armed bandits, policy iteration, and value iteration. The Frozen Lake enviroment was introduced in Chapter 2 of Grokking Reinforcement Learning by Miguel Morales.

## Files Overview

### Multi-Armed Bandit Implementations
- `SimpleBandits.ipynb`: Basic implementation of the multi-armed bandit problem with epsilon-greedy exploration strategies. Includes environment setup, bandit class implementation, and performance visualization.

- `AdvancedBandits.ipynb`: Enhanced version of the bandits implementation that adds linear and exponential epsilon decay strategies for improved exploration-exploitation balance.

- `TurboSimpleBandits.ipynb`: Simplified version focusing on core epsilon-greedy strategies without the additional complexity of decay or optimistic initialization.

### Frozen Lake Environment
- `FrozenLake.ipynb`: Implementation of policy evaluation and policy iteration algorithms for the Frozen Lake environment. Includes MDP definition with transition probabilities and rewards.

- `FrozenLake2.ipynb`: Extended implementation with value iteration algorithm and grid visualization for the Frozen Lake problem. Uses the same MDP structure but implements different solution approaches.

### Core Algorithm Implementations
- `ValueIterationAlgorithm.py`: Standalone implementation of the value iteration algorithm for solving MDPs.
- `frozen.py`: Core implementation of policy iteration algorithms for the Frozen Lake environment. Contains functions for policy evaluation, policy improvement, and the complete policy iteration process. Also includes visualization code to display the optimal policy using directional arrows.

## Key Concepts Covered

### Multi-Armed Bandits
- Basic epsilon-greedy strategy
- Optimistic initialization
- Different decay strategies:
  - Linear decay
  - Exponential decay
  - Constant epsilon

### Approximate Dynamic Programming Algorithms
- Policy Evaluation
- Policy Iteration
- Value Iteration
- State-value function computation
- Optimal policy extraction
