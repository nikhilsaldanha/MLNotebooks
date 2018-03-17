from blackjack import BlackjackEnv
import gym
from collections import defaultdict
import plotting
import sys
import numpy as np


def FirstVisitMCPolicyEvaluation(env, policy, num_episodes, gamma):

    R = defaultdict(float)
    N = defaultdict(float)
    V = defaultdict(float)

    # for each episode
    for _ in range(num_episodes):
        # get initial state
        state = env.reset()

        # keep track of states and rewards in the episode
        states, rewards = [], []

        # initialize done to False(game isn't over yet)
        done = False
        # sample actions from the policy until the game is done
        while not done:
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            states.append(state)
            rewards.append(reward)
            state = next_state

        visited_states = set(states)
        for s in visited_states:
            first_idx = states.index(s)
            G = np.sum(np.array(rewards[first_idx:]) * np.array([gamma ** i for i in range(0, len(rewards) - first_idx + 1)]))
            R[s] += G
            N[s] += 1.0
            V[s] = R[s] / N[s]

    return V


def EveryVisitMCPolicyEvaluation(env, policy, num_episodes, gamma):

    R = defaultdict(float)
    N = defaultdict(float)
    V = defaultdict(float)

    # for each episode
    for _ in range(num_episodes):
        # get initial state
        state = env.reset()

        # keep track of states and rewards in the episode
        states, rewards = [], []

        # initialize done to False(game isn't over yet)
        done = False
        # sample actions from the policy until the game is done
        while not done:
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            states.append(state)
            rewards.append(reward)
            state = next_state

        num_states = len(states)
        for i, s in enumerate(states):
            G = np.sum(np.array(rewards[i:]) * np.array([gamma ** i for i in range(0, num_states - i + 1)]))
            R[s] += G
            N[s] += 1.0
            V[s] = R[s] / N[s]

    return V

def IncrementalMCPolicyEvaluation(env, policy, num_episodes, gamma):

    N = defaultdict(float)
    V = defaultdict(float)

    # for each episode
    for _ in range(num_episodes):
        # get initial state
        state = env.reset()

        # keep track of states and rewards in the episode
        states, rewards = [], []
        
        # initialize done to False(game isn't over yet)
        done = False
        # sample actions from the policy until the game is done
        while not done:
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            states.append(state)
            rewards.append(reward)
            state = next_state
        
        num_states = len(states)
        
        for i, s in enumerate(states):
            G = np.sum(np.array(
                rewards[i:]) * np.array([gamma ** i for i in range(0, num_states - i)]))
            N[s] += 1.0
            V[s] = V[s] + 0.01 * (G - V[s])
    
    return V

def TDPolicyEvaluation(env, policy, num_episodes, gamma, alpha):

    # R = defaultdict(float)
    # N = defaultdict(float)
    V = defaultdict(float)

    # for each episode
    for _ in range(num_episodes):
        # get initial state
        state = env.reset()

        # keep track of states and rewards in the episode
        # states, rewards = [], []

        # initialize done to False(game isn't over yet)
        done = False
        # sample actions from the policy until the game is done
        while not done:
            # take a step
            action = policy(state)
            next_state, reward, done, _ = env.step(action)

            td_target = reward + gamma * V[next_state]
            td_delta = td_target - V[state]
            V[state] += alpha * td_delta

            # states.append(state)
            # rewards.append(reward)
            state = next_state

        # num_states = len(states)

        # for i, s in enumerate(states):
        #     G = np.sum(np.array(
        #         rewards[i:]) * np.array([gamma ** i for i in range(0, num_states - i)]))
        #     N[s] += 1.0
        #     V[s] = V[s] + 0.01 * (G - V[s])

    return V

def naive_policy(state):
    player_hand, _, _ = state
    if player_hand >= 20:
        return 0
    else:
        
        return 1

if __name__ == '__main__':
    env = BlackjackEnv()
    steps = 200000
    v = TDPolicyEvaluation(env, naive_policy, steps, 1.0, 0.5)
    # print(v)
    plotting.plot_value_function(v, title="{} Steps".format(steps))
