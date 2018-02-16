
import numpy as np
from gridworld import GridWorld
from policy_evaluation import policy_eval

def q_value(state, V, env, gamma):
    """
    Calculate the action value in a given state.
    
    Parameters:
    -----------
        state: int
            The state to calculate for which the value is to be calculated.

        V: numpy array (env.nS, )
            The value for each state

        env: GridWorld object
            The OpenAI Gym environment

        gamma: float
            The discount factor
    Returns:
    --------
        A: numpy array (env.nA, )
            Expected value of each action.
    """

    A = np.zeros(env.nA)
    for a in range(env.nA):
        for prob, next_state, reward, term in env.P[state][a]:
            A[a] += prob * (reward + gamma * V[next_state])
    return A

def value_iteration(env, gamma=1.0, theta=0.0001):
    """
    Value Iteration.

    Parameters:
    -----------
        env: GridWorld object
            The OpenAI Gym envrionment.

        gamma: float
            The discount factor.
        
        theta: float
            The threshold for delta improvement of state value

    Returns:
        policy: numpy array (S, A)
            The optimal policy
        
        V: numpy array (env.nS, )
            Value function for optimal policy
    """
    
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    V = np.zeros(env.nS)    
    while True:
        delta = 0
        for s in range(env.nS):
            # get best q value among all actions for state s
            action_values = q_value(s, V, env, gamma)
            
            # calculate delta
            delta = max(delta, abs(V[s] - max(action_values)))

            # update the q value as the state value
            V[s] = max(action_values)
        if delta < theta:
            break

    for s in range(env.nS):
        new_action_values = q_value(s, V, env, gamma)
        new_action = np.argmax(new_action_values)
        policy[s, new_action] = 1.0

    return policy, V

env = GridWorld((10, 10))
policy, v = value_iteration(env)
print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
print(np.reshape(np.argmax(policy, axis=1), env.shape))
