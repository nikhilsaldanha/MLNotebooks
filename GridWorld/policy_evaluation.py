import numpy as np
from gridworld import GridWorld


def policy_eval(policy, env, gamma=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.

    Parameters:
    -----------
        policy: numpy array (S, A)
            The initial policy.

        env: GridWorld object
            OpenAI Gym env.

        theta: float
            Threshold for delta in value function for stopping condition of policy evaluation.

        gamma: float
            Discount factor.

    Returns:
    --------
        V: numpy array (env.nS, )
            The value function.
    """

    V = np.zeros(env.nS)
    while True:
        delta = 0
        for s in range(env.nS):
            v = 0
            for a, action_prob in enumerate(policy[s]):
                for state_prob, next_state, reward, term in env.P[s][a]:
                    v += action_prob * state_prob * \
                        (reward + gamma * V[next_state])
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break
    return np.array(V)


if __name__ == '__main__':
    env = GridWorld()
    random_policy = np.ones([env.nS, env.nA]) / env.nA
    # print(random_policy)
    v = policy_eval(random_policy, env).reshape(env.shape)
    print(v)
