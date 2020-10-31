import numpy as np
import sys
import itertools

class QLearningAgent:
    '''
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy

    Attributes:
        action_size: Number of actions allowed in the environment.
        state_size: State space discretization
        gamma: Discount factor.
        alpha: TD learning rate.
        alpha_decay_rate
        min_alpha
        epsilon: Chance to sample a random action. Float between 0 and 1.
        epsilon_decay_rate TODO
        min_epsilon TODO
    '''

    def __init__(self, 
    action_size, 
    state_size, 
    **kargs):
        
        self.state_size = state_size
        self.action_size = action_size
        self.actions = np.arange(self.action_size)

        # Parameters
        self.gamma = kargs["gamma"]
        self.alpha = kargs["alpha"]
        self.min_alpha = kargs["min_alpha"]
        self.alpha_decay_rate = kargs["alpha_decay_rate"]
        self.epsilon = self.initial_epsilon = kargs["epsilon"]
        self.epsilon_decay_rate = kargs["epsilon_decay_rate"]
        self.min_epsilon = kargs["min_epsilon"]
        
        # Q Table, initialized with random Q values between -2 and 0
        shape = self.state_size + (self.action_size, )
        self.Q  = np.random.uniform(low=-2, high=0, size=shape)

        # Misc
        self.debug = False

    def policy(self, state):
        '''
        Epsilon greedy action selection
        '''

        # if len(set(self.Q[state])) == 1:
        #     A = np.ones(self.action_size, dtype=float) / self.action_size
        # else:
        #     A = np.ones(self.action_size, dtype=float) * self.epsilon / self.action_size
        #     best_action = np.argmax(self.Q[state])
        #     A[best_action] += (1.0 - self.epsilon)

        if np.random.random() < self.epsilon:
            chosen_action = np.random.choice(self.actions)
        else:
            # No tie-breaking
            # chosen_action = np.argmax(self.Q[state]) # No tie-breaking
            
            # Q-values to probabilities
            # logits = self.Q[state]
            # logits_exp = np.exp(logits)
            # probabilities = logits_exp / np.sum(logits_exp)
            # chosen_action = np.random.choice(self.actions, p=probabilities)

            # Tie-breaking
            values = self.Q[state]
            chosen_action = np.random.choice(np.flatnonzero(values == values.max()))
        return chosen_action

    def set_debug(self, flag=True):
        self.debug = flag

    def decay_epsilon(self, i=None):
        self.epsilon -= self.epsilon_decay_rate
        #self.epsilon = self.epsilon*self.epsilon_decay_rate
        #self.epsilon = np.exp(-self.epsilon_decay_rate*i)
        self.epsilon = max(self.epsilon, self.min_epsilon)

    def decay_alpha(self, i=None):
        self.alpha -= self.alpha_decay_rate
        self.alpha = max(self.alpha, self.min_alpha)

    def update(self, state, next_state, best_action, reward):
        '''
        Q table update
        '''

        t = state + (best_action, )
        qs = self.Q[next_state]
        td_target = reward + self.gamma * qs.max()
        td_error = td_target - self.Q[t]
        self.Q[t] += self.alpha * td_error

    def decide(self, state, env):
        '''
        Train for a single step
        '''
        action = self.policy(state)
        next_state, reward, done, _ = env.step(action)
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