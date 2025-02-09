# TorchRL Notebooks

## Contents

### Part1.ipynb
This notebook covers the fundamental concepts of TorchRL environments and their implementation. Key topics include:
- Creating and configuring environments using TorchRL's wrapper system
- Understanding the basics of TensorDict data structures for RL
- Working with environment methods like reset() and step()
- Using random actions and environment rollouts
- Implementing environment transforms with TransformedEnv
- Using the StepCounter transform for tracking episode progress

### Part2.ipynb
This notebook dives into more advanced topics, focusing on policies and networks in TorchRL. Topics covered include:
- Creating and implementing policy modules using TensorDictModule
- Working with specialized wrappers like Actor and ProbabilisticActor
- Understanding and implementing MLP and ConvNet networks
- Controlling action sampling and exploration strategies
- Implementing ε-greedy exploration
- Working with Q-Value actors for discrete action spaces
- Building and using value networks

The notebooks assume familiarity with:
- PyTorch
- Basic reinforcement learning concepts
