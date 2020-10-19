import numpy as np
import sys
import itertools
from collections import defaultdict

def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

class QLearning():
    '''
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy

    Attributes:
        num_actions: Number of actions allowed in the environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance to sample a random action. Float between 0 and 1.
    '''

    def __init__(self, num_actions, discount_factor=1.0, alpha=0.5, epsilon=0.1):
        # The final action-value function.
        # A nested dictionary that maps state -> (action -> action-value).
        self.Q = defaultdict(lambda: np.zeros(num_actions))
        # The policy we're following
        self.policy = make_epsilon_greedy_policy(self.Q, epsilon, num_actions)

        # Parameters
        self.discount_factor = discount_factor
        self.alpha = alpha
        self.epsilon = epsilon

        # Misc
        self.debug = False

    def set_debug(self, flag=True):
        self.debug = flag

    def update(self, state, next_state, action, reward):
        '''
        Q table update
        '''
        best_next_action = np.argmax(self.Q[next_state]) 
        td_target = reward + self.discount_factor * self.Q[next_state][best_next_action]
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] = self.Q[state][action] + self.alpha * td_error

    def step_train(self, state, env):
        '''
        Train for a single step
        '''
        action_probs = self.policy(state)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        next_state, reward, done, _ = env.step(action)

        # Next action
        next_action_probs = self.policy(next_state)
        next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)

        # Update statistics
        #stats.episode_rewards[i_episode] += reward
        #tats.episode_lengths[i_episode] = step

        # TD update
        best_next_action = np.argmax(Q[next_state]) 
        td_target = reward + self.discount_factor * self.Q[next_state][best_next_action]
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] = self.Q[state][action] + self.alpha * td_error

        state = next_state
        return next_state, done

    def train(self, env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
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
                action_probs = self.policy(state)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                next_state, reward, done, _ = env.step(action)

                # Next action
                next_action_probs = self.policy(next_state)
                next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)

                # Update statistics
                #stats.episode_rewards[i_episode] += reward
                #stats.episode_lengths[i_episode] = step

                # TD update
                best_next_action = np.argmax(self.Q[next_state]) 
                td_target = reward + self.discount_factor * self.Q[next_state][best_next_action]
                td_error = td_target - self.Q[state][action]
                self.Q[state][action] = self.Q[state][action] + self.alpha * td_error

                if done:
                    break
                state = next_state

