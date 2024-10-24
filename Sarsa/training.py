"""
Training utilities for SARSA(λ) agent.
Includes training loop and progress tracking.
"""

from typing import List
import numpy as np
from environment import Environment
from agent import SarsaLambda

def trainSarsaLambda(env: Environment, agent: SarsaLambda, 
                     nEpisodes: int = 100000, evalInterval: int = 1000) -> List[float]:
    """
    Train SARSA(λ) agent on given environment.
    
    Args:
        env: Environment to train in
        agent: SarsaLambda agent to train
        nEpisodes: Number of episodes to train for
        evalInterval: Interval for computing average reward
        
    Returns:
        List[float]: History of average rewards
    """
    rewardsHistory = []
    episodeRewards = []
    
    for episode in range(nEpisodes):
        state = env.reset()
        action = agent.chooseAction(state)
        episodeReward = 0
        
        while True:
            nextState, reward, isDone = env.step(action)
            episodeReward += reward
            
            if isDone:
                agent.update(state, action, reward, nextState, None, isDone)
                break
                
            nextAction = agent.chooseAction(nextState)
            agent.update(state, action, reward, nextState, nextAction, isDone)
            
            state, action = nextState, nextAction
            
        episodeRewards.append(episodeReward)
        agent.decayEpsilon(episode, nEpisodes)
        
        if episode % evalInterval == 0:
            avgReward = np.mean(episodeRewards[-evalInterval:])
            rewardsHistory.append(avgReward)
            print(f"Episode {episode}/{nEpisodes}, "
                  f"Average Reward: {avgReward:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}")
            
    return rewardsHistory