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

def policy_improvement(env, gamma=1.0):
    """
    Policy Improvement until optimality.    

    Parameters:
    -----------
        env: GridWorld object
            The OpenAI Gym envrionment.

        gamma: float
            The discount factor.
        
    Returns:
        policy: numpy array (S, A)
            The optimal policy
        
        V: numpy array (env.nS, )
            Value function for optimal policy
    """
    
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    
    while True:
        V = policy_eval(policy, env, gamma)
        
        converged = True

        for s in range(env.nS):
            old_action = np.argmax(policy[s])

            action_values = q_value(s, V, env, gamma)
            new_action = np.argmax(action_values)

            policy[s] = np.eye(env.nA)[new_action]

            if new_action > old_action:
                converged = False

        if converged:
            return policy, V

env = GridWorld((10, 10))
policy, v = policy_improvement(env)
print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
print(np.reshape(np.argmax(policy, axis=1), env.shape))
