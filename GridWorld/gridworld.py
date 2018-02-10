import numpy as np
import sys
from gym.envs.toy_text import discrete

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class GridWorld(discrete.DiscreteEnv):
    """
    An agent 'x' is in a MxN world with a goal to reach terminal states 'T'.
    The agent can move in each of the 4 cardinal directions, UP, DOWN, RIGHT, and LEFT.
    A reward of -1 is given to the agent after each action until it reaches a terminal state.

    Example grid world:
    T o o o
    o o o o
    o x o o
    o o o T
    """

    def __init__(self, shape=[4, 4]):
        """
        shape: list
            Shape of the 2D grid world.
            First index corresponds to number of rows.
            Second index corresponds to number of columns.
        """
        self.shape = shape
        nS = np.prod(self.shape)
        nA = 4
        isd = np.ones(nS) / nS

        grid = np.arange(nS).reshape(self.shape)
        it = np.nditer(grid, flags=['multi_index'])

        P = {s: {a: [] for a in range(nA)} for s in range(nS)}
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index
            # probability, nextstate, reward, done

            def is_term(s): return s == 0 or s == (nS - 1)
            reward = 0.0 if is_term(s) else -1.0

            if is_term(s) == True:
                s_up = s_right = s_down = s_left = s
                term_up = term_right = term_down = term_left = True
            else:
                s_up = s if y == 0 else s - self.shape[1]
                term_up = is_term(s_up)

                s_right = s if x == self.shape[1] - 1 else s + 1
                term_right = is_term(s_right)

                s_down = s if y == self.shape[0] - 1 else s + self.shape[1]
                term_down = is_term(s_down)

                s_left = s if x == 0 else s - 1
                term_left = is_term(s_left)

            P[s][UP] = [(1.0, s_up, reward, term_up)]
            P[s][RIGHT] = [(1.0, s_right, reward, term_right)]
            P[s][DOWN] = [(1.0, s_down, reward, term_down)]
            P[s][LEFT] = [(1.0, s_left, reward, term_left)]

            it.iternext()

        super(GridWorld, self).__init__(nS, nA, P, isd)

    def render(self, mode='human', close=False):
        if close == True:
            return

        outfile = StringIO() if mode == 'ansi' else sys.stdout

        grid = np.arange(self.nS).reshape(self.shape)
        it = np.nditer(grid, flags=['multi_index'])
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            if self.s == s:
                output = " x "
            elif s == 0 or s == self.nS - 1:
                output = " T "
            else:
                output = " o "

            if x == 0:
                output = output.lstrip()
            if x == self.shape[1] - 1:
                output = output.rstrip()

            outfile.write(output)

            if x == self.shape[1] - 1:
                outfile.write("\n")

            it.iternext()
