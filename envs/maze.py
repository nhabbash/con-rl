import io
import numpy as np
import sys
from gym.envs.toy_text import discrete

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class MazeEnv(discrete.DiscreteEnv):

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, 
                shape=[6, 9], 
                start=[2, 0], 
                goal=[0, 8],
                blocks = [11, 20, 29, 7, 16, 25, 41]):
        if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
            raise ValueError('shape argument must be a list/tuple of length 2')

        self.shape = shape
        self.blocks = blocks
        self.goal = goal
        self.start = start

        nS = np.prod(shape)
        nA = 4

        MAX_Y = shape[0]
        MAX_X = shape[1]

        P = {}
        grid = np.arange(nS).reshape(shape)

        goal_i = np.ravel_multi_index(goal, shape)
        blocks_i = np.array(np.unravel_index(blocks, shape)).transpose()
        it = np.nditer(grid, flags=['multi_index'])

        collision = lambda y, x: [y, x] in blocks_i.tolist()

        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            is_done = lambda ns: ns == goal_i
            reward = lambda ns: 1.0 if is_done(ns) else 0.0
        
            # P[s][a] = (prob, next_state, reward, is_done)
            P[s] = {a : [] for a in range(nA)}
     
            ns_up = s if y == 0 or collision(y-1, x) else s - MAX_X
            ns_right = s if x == (MAX_X - 1) or collision(y, x+1) else s + 1
            ns_down = s if y == (MAX_Y - 1) or collision(y+1, x) else s + MAX_X
            ns_left = s if x == 0 or collision(y, x-1) else s - 1
                
            P[s][UP] = [(1.0, ns_up, reward(ns_up), is_done(ns_up))]
            P[s][RIGHT] = [(1.0, ns_right, reward(ns_right), is_done(ns_right))]
            P[s][DOWN] = [(1.0, ns_down, reward(ns_down), is_done(ns_down))]
            P[s][LEFT] = [(1.0, ns_left, reward(ns_left), is_done(ns_left))]

            it.iternext()

        # Starting state
        isd = np.zeros(nS)
        isd[np.ravel_multi_index(self.start, self.shape)] = 1.0

        # We expose the model of the environment for educational purposes
        # This should not be used in any model-free learning algorithm
        self.P = P

        super(MazeEnv, self).__init__(nS, nA, P, isd)

    def render(self, mode='human', close=False):
        self._render(mode, close)

    def _render(self, mode='human', close=False):
        """ Renders the current gridworld layout
         For example, a 4x4 grid with the mode="human" looks like:
            T  o  o  o
            o  x  o  o
            o  o  o  o
            o  o  o  T
        where x is your position and T are the two terminal states.
        """
        if close:
            return

        outfile = io.StringIO() if mode == 'ansi' else sys.stdout

        grid = np.arange(self.nS).reshape(self.shape)
        it = np.nditer(grid, flags=['multi_index'])

        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            if self.s == s:
                output = " x "
            elif [y, x] == self.goal:
                output = " G "
            elif [y, x] == self.start:
                output = " S "
            elif s in self.blocks:
                output = " | "
            else:
                output = " . "

            if x == 0:
                output = output.lstrip()
            if x == self.shape[1] - 1:
                output = output.rstrip()

            outfile.write(output)

            if x == self.shape[1] - 1:
                outfile.write("\n")

            it.iternext()