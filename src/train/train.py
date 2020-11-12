from ..conrl import ConRL
from ..utils import *

import time
import sys
from collections import defaultdict
import itertools

import numpy as np
import pandas as pd
import wandb
import gym

env = gym.make('MountainCar-v0')
state_size = (10, 10)
window_size = (env.observation_space.high - env.observation_space.low)/state_size
num_episodes = 500
max_step = 1000
env._max_episode_steps = max_step

q_params = {
    "gamma": 0.9,
    "alpha": 0.1,
    "alpha_decay_rate": 0,
    "min_alpha": 0.1,
    "epsilon": 1.0,
    "epsilon_decay_rate": 0,
    "min_epsilon": 0.1
}

q_params["epsilon_decay_rate"] = (q_params["epsilon"] - q_params["min_epsilon"])/(num_episodes//2)
q_params["alpha_decay_rate"] = (q_params["alpha"] - q_params["min_alpha"])/(num_episodes//2)

mlgng_params = {
    "ndim": 2, 
    "e_w":0.5, 
    "e_n":0.1, 
    "l":10, 
    "a":0.5, 
    "b":0.95,
    "k":1000.0, 
    "max_nodes": 10, 
    "max_age": 10
}

stats_cr= {
        "episode_lengths":  np.zeros(num_episodes),
        "episode_rewards":  np.zeros(num_episodes),
        "selector_dist":    np.zeros((num_episodes, max_step)).astype(int),
        "mlgng_nodes":      [],
        "best_actions":     [],
}

wandb.init(
    entity="dodicin",
    project="con-rl",
    notes="test",
    tags=["q-learning", "mlgng"],
    config={"q_params": q_params,
            "mlgng_params": mlgng_params})

config = wandb.config

conrl = ConRL(action_size=env.action_space.n, state_size=state_size, update_threshold=10)
conrl.init_support(**config.q_params)
conrl.init_mlgng(**config.mlgng_params)

for episode in range(num_episodes):
    done = False
    success = False
    step = 0
    cumulative_reward = 0
    selector_dist_ep = np.zeros(max_step)

    start = time.time()
    obs = env.reset()

    state = get_discrete_state(obs, window_size, env)
    while not done:
        next_state, reward, done, selected = conrl.step(state, env, window_size=window_size, discretize=get_discrete_state)
        state = next_state
        
        cumulative_reward += reward
        selector_dist_ep[step] = selected

        step+=1
        if step >= max_step:
            break

    stats_cr["selector_dist"][episode] = selector_dist_ep
    stats_cr["episode_rewards"][episode] = cumulative_reward
    stats_cr["episode_lengths"][episode] = step

    wandb.log({
        'reward': stats_cr["episode_rewards"][episode], 
        'steps': stats_cr["episode_lengths"][episode],
        'selector': np.mean(stats_cr["selector_dist"][episode]),
        'global_error': conrl.mlgng.get_last_stat_tuple("global_error"),
        'node_number': np.sum([len(conrl.mlgng[i].g.get_vertices()) for i in range(conrl.mlgng.m)])
        })
        
    conrl.support.decay_epsilon(episode)
    end = time.time() - start
    if episode % 100 == 0:
        print("Episode {}/{}, Reward {}, Total steps {}, Epsilon: {:.2f}, Alpha: {:.2f}, Time {:.3f}".format(
            episode, 
            num_episodes, 
            stats_cr["episode_rewards"][episode], 
            stats_cr["episode_lengths"][episode], 
            conrl.support.epsilon, 
            conrl.support.alpha, 
            end))
        conrl.mlgng.print_stats(one_line=True)