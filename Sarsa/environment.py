"""
Environment class for handling MDP transitions and state management.
Provides an interface for reinforcement learning agents to interact with the environment.
"""

import random
from typing import List, Tuple, Dict, Any

class Environment:
    def __init__(self, mdp: Dict[int, Dict[int, List[Tuple[float, int, float, bool]]]]):
        """
        Initialize environment with a Markov Decision Process.
        
        Args:
            mdp: Dictionary mapping states to actions to transitions
                {state: {action: [(probability, nextState, reward, isDone)]}}
        """
        self.mdp = mdp
        self.states = list(mdp.keys())
        self.currentState = random.choice(self.states)

    def reset(self) -> int:
        """
        Reset environment to random initial state.
        
        Returns:
            int: Initial state ID
        """
        self.currentState = random.choice(self.states)
        return self.currentState

    def step(self, action: int) -> Tuple[int, float, bool]:
        """
        Take an action in the current state.
        
        Args:
            action: Action ID to take
            
        Returns:
            Tuple containing:
                - Next state ID
                - Reward received
                - Whether episode is done
                
        Raises:
            ValueError: If action is invalid for current state
        """
        if action in self.mdp[self.currentState]:
            transitions = self.mdp[self.currentState][action]
            probabilities = [t[0] for t in transitions]
            nextStateIndex = random.choices(range(len(transitions)), weights=probabilities)[0]
            probability, nextState, reward, isDone = transitions[nextStateIndex]
            self.currentState = nextState
            return nextState, reward, isDone
        else:
            raise ValueError(f"Invalid action {action} for state {self.currentState}")

    def isTerminal(self, state: int) -> bool:
        """
        Check if given state is terminal.
        
        Args:
            state: State ID to check
            
        Returns:
            bool: True if state is terminal
        """
        return any(transition[3] for action in self.mdp.get(state, {}) 
                  for transition in self.mdp[state][action])
