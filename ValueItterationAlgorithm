import numpy as np

def value_iteration(P, gamma=1.0, theta=1e-10):
    """
    Value Iteration algorithm to compute the optimal state-value function and policy.
    
    Parameters:
    P : list
        Transition probability matrix for the MDP.
        P[s][a] is a list of (probability, next_state, reward, done) tuples.
    gamma : float
        Discount factor.
    theta : float
        Convergence threshold.
    
    Returns:
    V : np.ndarray
        Optimal state-value function.
    pi : np.ndarray
        Optimal policy.
    """
    V = np.zeros(len(P), dtype=np.float64)  # (1) Initialize state-value function

    while True:  # (2) Loop until convergence
        Q = np.zeros((len(P), len(P[0])), dtype=np.float64)  # (3) Initialize action-value function/Q function

        for s in range(len(P)):  # (4) Iterate over all states
            for a in range(len(P[s])):  # Iterate over all actions for each state
                for prob, next_state, reward, done in P[s][a]:  # (5) Iterate over transitions
                    # (6) Calculate the Q-value for each action
                    #A version of the Bellman equation
                    Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))
        
        # (7) Check if the state-value function has converged
        if np.max(np.abs(V - np.max(Q, axis=1))) < theta:
            break
        
        # (8) Update the state-value function
        V = np.max(Q, axis=1)
        # (9) Extract the optimal policy
        pi = np.argmax(Q, axis=1)   


    return V, pi
