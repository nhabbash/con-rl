import numpy as np
import time

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
                **kwargs):
        
        self.state_size = state_size
        self.action_size = action_size
        self.actions = np.arange(self.action_size)
        self.exploitation = False
        # Parameters

        if kwargs:
            self.gamma = kwargs["gamma"]
            self.alpha = kwargs["alpha"]
            self.min_alpha = kwargs["min_alpha"]
            self.alpha_decay_rate = kwargs["alpha_decay_rate"]
            self.epsilon = self.initial_epsilon = kwargs["epsilon"]
            self.epsilon_decay_rate = kwargs["epsilon_decay_rate"]
            self.min_epsilon = kwargs["min_epsilon"]
        
        # Q Table, initialized with random Q values between -2 and 0
        shape = self.state_size + (self.action_size, )
        self.Q  = np.random.uniform(low=-2, high=0, size=shape)

        # Misc
        self.debug = False

    def set_parameters(self, **kwargs):
            self.gamma = kwargs["gamma"]
            self.alpha = kwargs["alpha"]
            self.min_alpha = kwargs["min_alpha"]
            self.alpha_decay_rate = kwargs["alpha_decay_rate"]
            self.epsilon = self.initial_epsilon = kwargs["epsilon"]
            self.epsilon_decay_rate = kwargs["epsilon_decay_rate"]
            self.min_epsilon = kwargs["min_epsilon"]

    def policy(self, state, exploitation=False):
        '''
        Epsilon greedy action selection
        '''

        # if len(set(self.Q[state])) == 1:
        #     A = np.ones(self.action_size, dtype=float) / self.action_size
        # else:
        #     A = np.ones(self.action_size, dtype=float) * self.epsilon / self.action_size
        #     best_action = np.argmax(self.Q[state])
        #     A[best_action] += (1.0 - self.epsilon)

        if np.random.random() < self.epsilon and not exploitation:
            chosen_action = np.random.choice(self.actions)
        else:
            # Q-values to probabilities
            # logits = self.Q[state]
            # logits_exp = np.exp(logits)
            # probabilities = logits_exp / np.sum(logits_exp)
            # chosen_action = np.random.choice(self.actions, p=probabilities)

            # Random tie-breaking
            values = self.Q[state]
            chosen_action = np.random.choice(np.flatnonzero(values == values.max()))
        return chosen_action

    def set_debug(self, flag=True):
        self.debug = flag

    def decay_param(self, param):
        decay = getattr(self, param+"_decay_rate", 0)
        value = getattr(self, param, 0)
        min_value = getattr(self, "min_"+param, 0)
        setattr(self, param, max(value-decay, min_value))
        # self.epsilon -= self.epsilon_decay_rate
        # #self.epsilon = self.epsilon*self.epsilon_decay_rate
        # #self.epsilon = np.exp(-self.epsilon_decay_rate*i)
        # self.epsilon = max(self.epsilon, self.min_epsilon)

    def update(self, state, next_state, best_action, reward):
        '''
        Q table update
        '''

        t = state + (best_action, )
        qs = self.Q[next_state]
        td_target = reward + self.gamma * qs.max()
        td_error = td_target - self.Q[t]
        self.Q[t] += self.alpha * td_error

    def step(self, state, env):
        if not isinstance(state, tuple):
            state = (state, )
        action = self.policy(state)
        next_state, reward, done, _ = env.step(action)
        self.update(state, next_state, action, reward)
        state = next_state

        return next_state, reward, done

    def train(self, env, num_episodes, stats):

        for episode in range(num_episodes):
            done = False
            step = 0
            cumulative_reward = 0

            start = time.time()
            state = env.reset()

            while not done:
                next_state, reward, done = self.step(state, env)
                state = next_state
                step+=1
                cumulative_reward+=reward
            self.decay_param("epsilon")

            stats["cumulative_reward"][episode] = cumulative_reward
            stats["step"][episode] = step 
            stats["q_tables"][episode] = self.Q
            stats["best_actions"].append(self.get_best_actions())

            end = time.time() - start
            if (episode+1) % 50 == 0:
                print("Episode {}/{}, Reward {}, Average Max Reward: {}, Total steps {}, Epsilon: {:.2f}, Alpha: {:.2f}, Time {:.3f}".format(
                    episode+1, 
                    num_episodes, 
                    stats["cumulative_reward"][episode],
                    stats["cumulative_reward"][episode-10:episode].mean(),
                    stats["step"][episode], 
                    self.epsilon, 
                    self.alpha,
                    end))

    def get_best_actions(self):
        best_actions_table = np.argmax(self.Q, axis=len(self.state_size))
        length = np.prod(self.state_size)
        best_actions = np.empty((length, len(self.state_size)+1))

        for idx in range(length):
            state = np.unravel_index(idx, self.state_size)
            best_a = best_actions_table[state]
            best_actions[idx] = state + (best_a, )

        return best_actions.T