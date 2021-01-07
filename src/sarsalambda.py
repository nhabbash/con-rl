import numpy as np
import time

class SarsaLambdaAgent:
    '''
    
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
            self.trace_decay = kwargs["trace_decay"]
            self.gamma = kwargs["gamma"]
            self.alpha = kwargs["alpha"]
            self.min_alpha = kwargs["min_alpha"]
            self.alpha_decay_rate = kwargs["alpha_decay_rate"]
            self.epsilon = self.initial_epsilon = kwargs["epsilon"]
            self.epsilon_decay_rate = kwargs["epsilon_decay_rate"]
            self.min_epsilon = kwargs["min_epsilon"]
        
        # Q Table
        shape = self.state_size + (self.action_size, )
        self.Q  = np.random.uniform(low=0, high=0, size=shape)
        # Eligibility traces table
        self.E = np.random.uniform(low=0, high=0, size=shape)

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
        if np.random.random() < self.epsilon and not exploitation:
            chosen_action = np.random.choice(self.actions)
        else:
            chosen_action = np.argmax(self.Q[state])

        return chosen_action

    def set_debug(self, flag=True):
        self.debug = flag

    def decay_param(self, param):
        decay = getattr(self, param+"_decay_rate", 0)
        value = getattr(self, param, 0)
        min_value = getattr(self, "min_"+param, 0)
        setattr(self, param, max(value-decay, min_value))

    def update(self, state, next_state, action, reward, type="accumulate"):
        '''
        Q table+Eligibility traces update
        '''
        next_action = self.policy(next_state)
        delta = reward + self.gamma * self.Q[next_state][next_action] - self.Q[state][action]

        self.E[state][action] += 1

        self.Q += self.alpha * delta * self.E 
        if type == 'accumulate':
            self.E *= self.gamma * self.trace_decay
        elif type == 'replace':
            self.E *= self.gamma * self.trace_decay  
            self.E[state] = 1

        return next_action        

    def step(self, state, action, env):
        if not isinstance(state, tuple):
            state = (state, )
        next_state, reward, done, _ = env.step(action)

        return next_state, reward, done

    def train(self, env, num_episodes, stats):

        for episode in range(num_episodes):
            done = False
            step = 0
            cumulative_reward = 0

            start = time.time()
            state = env.reset()
            action = self.policy(state)

            while not done:
                next_state, reward, done = self.step(state, action, env)

                next_action = self.update(state, next_state, action, reward)

                state = next_state
                action = next_action

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

    def evaluate(self, env, stats, episode=0):
            done = False
            step = 0
            cumulative_reward = 0

            start = time.time()
            state = env.reset()

            while not done:
                if not isinstance(state, tuple):
                    state = (state, )

                action = self.policy(state, exploitation=True)
                next_state, reward, done, _ = env.step(action)
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
                print("#### QL: Episode {}, Reward {}, Average Max Reward: {}, Total steps {}, Epsilon: {:.2f}, Alpha: {:.2f}, Time {:.3f}".format(
                    episode+1,
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