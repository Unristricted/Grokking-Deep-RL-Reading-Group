
import numpy as np
import random
from frozenmdp import MDP
def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    prev_V = np.zeros(len(P))

    while True:
        V = np.zeros(len(P))
        for s in range(len(P)):
            for prob, next_state, reward, done in P[s][pi(s)]:
                V[s] += prob * (reward + gamma * prev_V[next_state] * (not done))
        if np.max(np.abs(prev_V - V)) < theta:
            break
        prev_V = V.copy()
    return V

def policy_improvement(V, P, gamma=1.0):
    Q = np.zeros((len(P), len(P[0])), dtype=np.float64)

    for s in range(len(P)):
        for a in range(len(P[s])):
            for prob, next_state, reward, done in P[s][a]:
                Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))
    new_pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
    return new_pi

def policy_iteration(P, gamma=1.0, theta = 1e-10):
    random_actions = np.random.choice(tuple(P[0].keys()), len(P))
    pi = lambda s: {s:a for s, a in enumerate(random_actions)}[s]

    while True:
        old_pi = {s:pi(s) for s in range(len(P))}
        V = policy_evaluation(pi, P, gamma, theta)
        pi = policy_improvement(V, P, gamma)

        if old_pi == {s:pi(s) for s in range(len(P))}: break
    return V, pi

values, myPolicy = policy_iteration(MDP)
print(values)

translate = {0: "<", 1: "v", 2: ">", 3:"^"}

R, C = 4,4
for i in range(R):
    row = []
    for j in range(C):
        state = (i*C)+j
        if values[state] == 0.0:
            row.append("O")
        else:
            row.append(translate[myPolicy((i*C)+j)])
    print(*row)
