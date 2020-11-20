from ..conrl import ConRL
from ..utils import *

import time

import numpy as np
import pandas as pd
import wandb
import gym

state_size = (10, 10)
env = DiscretizationWrapper(gym.make('MountainCar-v0'), state_size)

num_episodes = 500
max_step = 1000
env.env._max_episode_steps = max_step
env.spec.max_episode_steps = max_step

q_params = {
    "gamma": 0.9,
    "alpha": 0.1,
    "alpha_decay_rate": 0,
    "min_alpha": 0.1,
    "epsilon": 0.9,
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
    "b":0.6,
    "k":1000.0, 
    "max_nodes": 10, 
    "max_age": 10
}

stats = {
        "step":  np.zeros(num_episodes),
        "cumulative_reward":  np.zeros(num_episodes),
        "selector":    np.zeros(num_episodes),
        "mlgng_nodes":      [],
        "best_actions":     [],
}

wandb.init(
    entity="dodicin",
    project="con-rl",
    notes="test2",
    tags=["q-learning", "mlgng"])

config = wandb.config

conrl = ConRL(action_size=env.action_space.n, state_size=state_size, update_threshold=10)
conrl.init_support(**q_params)
conrl.init_mlgng(**mlgng_params)

for episode in range(num_episodes):
    done = False
    step = 0
    cumulative_reward = 0
    selector_sequence = []

    start = time.time()
    state = env.reset()

    while not done:
        next_state, reward, done, selected = conrl.step(state, env)
        state = next_state
        
        cumulative_reward += reward
        selector_sequence.append(selected)
        step+=1

    conrl.support.decay_param("epsilon")

    stats["selector"][episode] = sum(selector_sequence)/len(selector_sequence)
    stats["cumulative_reward"][episode] = cumulative_reward
    stats["step"][episode] = step 
    stats["best_actions"].append(conrl.get_best_actions())
    stats["mlgng_nodes"].append(conrl.mlgng.get_nodes())

    wandb.log({
        'cumulative_reward': stats["episode_rewards"][episode], 
        'step': stats["step"][episode],
        'selector': stats["selector"][episode],
        'global_error': conrl.mlgng.get_last_stat_tuple("global_error"),
        'node_number': [len(conrl.mlgng[i].g.get_vertices()) for i in range(conrl.mlgng.m)]
    })
        
    end = time.time() - start
    if (episode+1) % 50 == 0:
        print("Episode {}/{}, Reward {}, Average Max Reward: {}, Total steps {}, Epsilon: {:.2f}, Alpha: {:.2f}, Time {:.3f}".format(
            episode+1, 
            num_episodes, 
            stats["cumulative_reward"][episode],
            stats["cumulative_reward"][episode-10:episode].mean(),
            stats["step"][episode], 
            conrl.support.epsilon, 
            conrl.support.alpha, 
            end))
        conrl.mlgng.print_stats(one_line=True)
