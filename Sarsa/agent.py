"""
SARSA(λ) agent implementation with eligibility traces.
Includes epsilon-greedy action selection and experience replay.
"""

import random
import numpy as np
from typing import Optional, List

class SarsaLambda:
    def __init__(self, nStates: int, nActions: int, alpha: float = 0.1, 
                 gamma: float = 0.99, epsilon: float = 0.01, lambd: float = 0.9):
        """
        Initialize SARSA(λ) agent.
        
        Args:
            nStates: Number of possible states
            nActions: Number of possible actions
            alpha: Learning rate
            gamma: Discount factor
            epsilon: Exploration rate
            lambd: Trace decay parameter
        """
        self.nStates = nStates
        self.nActions = nActions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.lambd = lambd
        
        self.qTable = np.zeros((nStates, nActions))
        self.eTable = np.zeros((nStates, nActions))
        self.episodeRewards: List[float] = []
        
    def chooseAction(self, state: int) -> int:
        """
        Choose action using epsilon-greedy policy.
        
        Args:
            state: Current state ID
            
        Returns:
            int: Chosen action ID
        """
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.nActions - 1)
        return int(np.argmax(self.qTable[state]))
        
    def update(self, state: int, action: int, reward: float, 
               nextState: int, nextAction: Optional[int], isDone: bool):
        """
        Update Q-values using SARSA(λ) algorithm.
        
        Args:
            state: Current state ID
            action: Taken action ID
            reward: Received reward
            nextState: Resulting state ID
            nextAction: Next chosen action ID (None if terminal)
            isDone: Whether episode is complete
        """
        if isDone:
            tdError = reward - self.qTable[state, action]
        else:
            tdError = (reward + self.gamma * self.qTable[nextState, nextAction] 
                      - self.qTable[state, action])
        
        self.eTable[state, action] += 1
        self.qTable += self.alpha * tdError * self.eTable
        self.eTable *= self.gamma * self.lambd
        
        if isDone:
            self.resetEligibilityTraces()
            
    def resetEligibilityTraces(self):
        """Reset eligibility traces to zero."""
        self.eTable.fill(0.0)

    def decayEpsilon(self, episode: int, totalEpisodes: int):
        """
        Decay exploration rate over time.
        
        Args:
            episode: Current episode number
            totalEpisodes: Total number of episodes
        """
        self.epsilon = max(0.01, self.epsilon * (1 - episode/totalEpisodes))
