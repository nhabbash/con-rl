import sys
sys.path.append('/home/nassim/dev/conrl')

from src.conrl import ConRL
from src.qlearning import QLearningAgent
from src.utils import *

import time

import numpy as np
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
    "min_epsilon": 0.01
}

q_params["epsilon_decay_rate"] = (q_params["epsilon"] - q_params["min_epsilon"])/(num_episodes//2)
q_params["alpha_decay_rate"] = (q_params["alpha"] - q_params["min_alpha"])/(num_episodes//2)

mlgng_params = {
    "ndim": 2, 
    "e_w":0.05, 
    "e_n":0.005, 
    "l":10, 
    "a":0.5, 
    "b":0.95,
    "k":1000.0, 
    "max_nodes": 5, 
    "max_age": 200,
    "min_error": 5,
    "node_multiplier": 10
}

stats = build_conrl_stats(num_episodes, env)

wandb.init(
    entity="dodicin",
    project="con-rl",
    notes="hyperparameter_tuning",
    tags=["q-learning", "mlgng"])

config = wandb.config

conrl = ConRL(action_size=env.action_space.n, state_size=state_size, update_threshold=10)
support = QLearningAgent(action_size=env.action_space.n, state_size=state_size, **q_params)
conrl.init_support(support)
conrl.init_mlgng(**mlgng_params)

print("#### Starting training #####")
conrl.rewards = stats["cumulative_reward"]
conrl.max_avg_reward = np.NINF
conrl.init_adaptive_lr_params()
print_freq=50

for episode in range(num_episodes):
    conrl.episode = episode
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

    stats["selector"][episode] = sum(selector_sequence)/len(selector_sequence)
    stats["cumulative_reward"][episode] = cumulative_reward
    stats["step"][episode] = step 
    stats["global_error"][episode] = conrl.mlgng.get_last_stat_tuple("global_error")
    stats["nodes"][episode] = conrl.mlgng.get_last_stat_tuple("vertices")
    stats["rate"][episode] = conrl.discount_selector()

    wandb.log({
    'cumulative_reward': stats["cumulative_reward"][episode], 
    'step': stats["step"][episode],
    'selector': stats["selector"][episode],
    'global_error': conrl.mlgng.get_last_stat_tuple("global_error"),
    'node_number': [len(conrl.mlgng[i].g.get_vertices()) for i in range(conrl.mlgng.m)]})

    if episode > 0:
        conrl.update_lr(episode)
        conrl.decay_param("discount", episode, decay_rate=0.015)
        conrl.support.epsilon = conrl.discount

    if episode >= print_freq:
        lower_bound = episode-print_freq
        higher_bound = episode
    else:
        lower_bound = 0
        higher_bound = episode+1
    
    mean_reward = conrl.rewards[lower_bound:higher_bound].mean()
    if mean_reward > conrl.max_avg_reward:
        conrl.max_avg_reward = mean_reward
    
    stats["max_avg_reward"][episode] = conrl.max_avg_reward

    if (episode+1) % print_freq == 0:
        end = time.time() - start
        print("Episode {}/{}, Average Reward: {:.2f}, Global Error: {:.2f}, Total steps {}, Discount: {:.2f}, Time {:.3f}".format(
            episode+1, 
            num_episodes, 
            stats["cumulative_reward"][lower_bound:higher_bound].mean(),
            stats["global_error"][episode].sum(),
            stats["step"][episode],
            conrl.discount,
            stats["rate"][episode], 
            end))
        conrl.mlgng.print_stats(one_line=True)

