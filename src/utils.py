from collections import namedtuple
import numpy as np
import gym

def build_conrl_stats(num_episodes, env):
    return {
            "step":  np.zeros(num_episodes),
            "cumulative_reward":  np.zeros(num_episodes),
            "selector":    np.zeros(num_episodes),
            "global_error":     np.zeros((num_episodes, env.action_space.n)),
            "mlgng_nodes":      [],
            "best_actions":     [],
            "nodes": np.zeros((num_episodes, env.action_space.n)),
            "rate": np.zeros(num_episodes),
            "max_avg_reward": np.zeros(num_episodes),
    }

def dict_to_namedtuple(dict):
    nt = namedtuple('nt', dict)
    tuple = nt(**dict)
    return tuple

def create_discretization_grid(low, high, bins=[10]):
    if len(bins) == 1:
        bins = np.repeat(bins, len(low))

    assert len(low) == len(high) == len(bins)

    grid = [np.linspace(low[i], high[i], bins[i], endpoint=False)[1:] for i, _ in enumerate(low)]
    return np.array(grid)

def get_discrete_state(state, window_size, env):
    discrete_state = (state - env.observation_space.low)/window_size
    return tuple(discrete_state.astype(np.int))

class DiscretizationWrapper(gym.ObservationWrapper):
    def __init__(self, env, state_size):
        super().__init__(env)
        self.state_size = state_size
        self.window_size = (env.observation_space.high - env.observation_space.low)/self.state_size
    
    def observation(self, obs):
        discrete_state = (obs - self.env.observation_space.low)/self.window_size
        discrete_state = discrete_state.astype(np.int)
        discrete_state = [np.clip(state, 0, self.state_size[idx]-1) 
        for idx, state in enumerate(discrete_state)]
        return tuple(discrete_state)