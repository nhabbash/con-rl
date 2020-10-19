import numpy as np
import gym

class KArmedBanditsEnv(gym.Env):
    """
    Bandit environment base to allow agents to interact with the class k-armed bandits as defined in Sutton and Barto's Reinforcement Learning: An Introduction - Chapter 2

    r_dist:
        The payoffs (rewards) of each lever. It's defined as a list of means and standard deviations of a normal distribution for each lever. (called q*(a))
    non_stat:
        If set to True, the reward distributions change over time by incrementing each q*(a) of a certain sampled amount from a normal distibution with mean=0 and stdev=0.01. This makes it so the environment is non-stationary.
    """
    metadata = {'render.modes': ['human', 'ansi', 'text']}

    def __init__(self, r_dist, non_stat=False):
        for reward in r_dist:
            if not isinstance(reward, list):
                raise ValueError("r_dist must be in the format [[mean, stdev], [mean, stdev], ...]")
            elif reward[1] <= 0:
                raise ValueError("Every stdev in the rewards has to be greater than 0")

        self.r_dist = r_dist
        self.k = len(r_dist)
        self.non_stat = non_stat
        self.action_space = gym.spaces.Discrete(self.k)
        self.observation_space = gym.spaces.Discrete(1) # Nonassociative Bandits

    def step(self, action):
        assert self.action_space.contains(action)

        reward = 0
        done = True

        if self.non_stat:
            # Non-stationary independent random walk of q*(a)
            inc = np.random.normal(0.0, 0.01, self.k)
            self.r_dist = [(mu+inc[idx], sigma) for idx, (mu, sigma) in enumerate(self.r_dist)]

        reward = np.random.normal(self.r_dist[action][0], self.r_dist[action][1])

        return 0, reward, done, {}
    
    def render(self, mode='human', close='False'):
        pass