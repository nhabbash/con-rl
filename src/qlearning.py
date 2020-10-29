import numpy as np
import sys
import itertools
from collections import defaultdict

def discretize(sample, grid):
    coords = [int(np.digitize(sample[i], grid[i])) for i, _ in enumerate(grid)]
    return tuple(coords)

class QLearningAgent:
    '''
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy

    Attributes:
        action_size: Number of actions allowed in the environment.
        num_episodes: Number of episodes to run for.
        gamma: Discount factor.
        alpha: TD learning rate.
        epsilon: Chance to sample a random action. Float between 0 and 1.
        epsilon_decay_rate TODO
        min_epsilon TODO
        state_grid: State space discretization
    '''

    def __init__(self, action_size, state_grid, gamma=0.9, alpha=0.1, epsilon=0.1, epsilon_decay_rate=1, min_epsilon=0.01, seed=42):
        
        self.state_grid = state_grid
        self.state_size = tuple(len(dim) + 1 for dim in state_grid)
        self.action_size = action_size
        self.seed = np.random.seed(seed)

        # Parameters
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = self.initial_epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.min_epsilon = min_epsilon

        # Q Table
        shape = self.state_size + (self.action_size, )
        self.Q = np.zeros(shape=shape)
        # A nested dictionary that maps state -> (action -> action-value).
        #self.Q_old = defaultdict(lambda: np.zeros(action_size))

        # Misc
        self.debug = False

    def policy(self, state):
        '''
        Epsilon greedy action selection
        '''

        if len(set(self.Q[state])) == 1:
            A = np.ones(self.action_size, dtype=float) / self.action_size
        else:
            A = np.ones(self.action_size, dtype=float) * self.epsilon / self.action_size
            best_action = np.argmax(self.Q[state])
            A[best_action] += (1.0 - self.epsilon)

        chosen_action = np.random.choice(np.arange(len(A)), p=A)
        return chosen_action

    def set_debug(self, flag=True):
        self.debug = flag

    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay_rate
        self.epsilon = max(self.epsilon, self.min_epsilon)

    def update(self, state, next_state, best_action, reward):
        '''
        Q table update
        '''
        t = state + (best_action, )
        qs = self.Q[next_state]
        td_target = reward + self.gamma * max(qs)
        td_error = td_target - self.Q[t]
        self.Q[t] += self.alpha * td_error

        # best_next_action = np.argmax(self.Q[next_state]) 
        # td_target = reward + self.gamma * self.Q[next_state][best_next_action]
        # td_error = td_target - self.Q[state][best_action]
        # self.Q[state][best_action] = self.Q[state][best_action] + self.alpha * td_error

    def decide(self, state, env):
        '''
        Train for a single step
        '''
        action = self.policy(state)
        next_state, reward, done, _ = env.step(action)

        # TD update
        self.update(state, next_state, action, reward)

        state = next_state
        return next_state, done

    def train(self, env, num_episodes):
        '''
        Train for a num_episodes
        '''
        
        for i_episode in range(num_episodes):
            # Print out which episode we're on, useful for debugging.
            if (i_episode + 1) % 100 == 0 and self.debug:
                print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
                sys.stdout.flush()
            
            # Generate episode
            state = env.reset()
            for step in itertools.count():
                next_state, done = self.decide(state, env)

                if done:
                    break
                state = next_state