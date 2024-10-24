"""
Visualization utilities for reinforcement learning results.
Includes training progress and Q-value visualization.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List

def plotTrainingProgress(rewardsHistory: List[float], evalInterval: int):
    """
    Plot training progress over time.
    
    Args:
        rewardsHistory: List of average rewards
        evalInterval: Interval between evaluations
    """
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(rewardsHistory)) * evalInterval, rewardsHistory)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Training Progress')
    plt.grid(True)
    plt.show()

def plotQValueHeatmap(qTable: np.ndarray, gridSize: int = 4):
    """
    Plot heatmap of maximum Q-values for each state.
    
    Args:
        qTable: Q-value table
        gridSize: Size of the environment grid
    """
    maxQValues = np.max(qTable, axis=1).reshape(gridSize, gridSize)
    plt.figure(figsize=(8, 6))
    plt.imshow(maxQValues, cmap='coolwarm')
    plt.colorbar(label='Max Q-Value')
    for i in range(gridSize):
        for j in range(gridSize):
            plt.text(j, i, f'{maxQValues[i, j]:.2f}', 
                    ha='center', va='center')
    plt.title('Max Q-Values Across States')
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.show()