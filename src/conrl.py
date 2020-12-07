from .mlgng import MultiLayerGrowingNeuralGas
import numpy as np
import time

class ConRL():
    '''
    Constructivist Reinforcement Learning
    from Gueriau et al, 2019

    Attributes:
        support (): Supporting agent
        mlgng (): MLGNG agent
        action_counter (dict of dict): A nested dictionary that maps state -> (action -> counter).
        ndim (int): State space dimensionality
        update_threshold (int): Update threshold for each action layer of MLGNG
    @author: Nassim Habbash
    '''

    def __init__(self, action_size, state_size, update_threshold = 20, discount = 1.0):
        self.action_size = action_size
        self.state_size = state_size
        self.discount = discount
        self.episode = 0

        shape = state_size + (self.action_size, )
        self.action_counter = np.zeros(shape=shape)
        self.update_threshold = update_threshold
        
    def init_support(self, support):
        self.support = support

    def init_mlgng(self, **params):
        self.mlgng = MultiLayerGrowingNeuralGas(m=self.action_size, ndim=params["ndim"])
        self.mlgng.set_layers_parameters(params, m=-1)

    def init_adaptive_lr_params(self):
        self.m = 0
        self.v = 0
        self.lr = 1
        self.alpha = 1
        self.epsilon = 10e-8
        self.beta1 = 0.9
        self.beta2 = 0.9999

    def update_lr(self, t):

        #g = self.rewards[self.episode]-self.rewards[self.episode-1]
        g = (self.rewards[self.episode]-self.rewards[self.episode-2])/2
        self.m = self.beta1*self.m+(1-self.beta1)*g
        self.v = self.beta2*self.v+(1-self.beta2)*g**2
        m_s = self.m/(1-self.beta1**t)
        v_s = self.v/(1-self.beta2**t)
        self.lr = self.alpha*m_s/(np.sqrt(v_s)+self.epsilon)
    
    def decay_param(self, param, episode, decay_rate=0.015, func=lambda x, y: np.exp(-x*y)):
        setattr(self, param, func(decay_rate, episode))
        
    def simple_action_selector(self, state):
        '''
        Action selection, if MLGNG has an action, select it, otherwise get it from QL

        Parameters:
        state (tuple): Sampled state
        '''

        # QL action - choose the action with the highest Q value with epsilon probability to explore
        q_best_action = self.support.policy(state)

        # MLGNG action - choose the action layer with the closest state
        mlgng_best_action = self.mlgng.policy(state)

        best_action = None
        selected = None
        if mlgng_best_action is None:
            best_action = q_best_action
            selected = 0 # Support
        else: 
            best_action = mlgng_best_action
            selected = 1 # MLGNG

        return best_action, mlgng_best_action, q_best_action, selected
        
    def mlgng_update_strategy(self, state, support_best_action):
        '''
        MLGNG is updated only after a state reaches a threshold of consecutive actions from the supporting agent
        '''

        # Consecutive action counter update
        # If the counter for an incoming (state, action) pair is 0, 
        # reset the counter for all state's actions to start a new consecutive count
        if self.action_counter[state][support_best_action] == 0:
            self.action_counter[state] = 0
        
        self.action_counter[state][support_best_action] += 1

        # MLGNG Agent update

        if self.action_counter[state][support_best_action] >= self.update_threshold:
            self.mlgng.update(state, support_best_action)
            self.mlgng.update_discount_rate(np.abs(self.lr))
            self.action_counter[state] = 0

    def discount_selector(self):
        if self.episode >= 10:
            lower_window = self.episode-10
        else:
            lower_window = 0

        # If low variance (flat area of the reward) and low reward compared to the past, or LR negative (reward is falling down), use the exponential discount
        #
        #     np.mean(self.rewards[lower_window:self.episode]) <= self.max_avg_reward) or self.lr < 0:
        if np.std(self.rewards[lower_window:self.episode]) <= 50:
            return self.discount
        else:
            return self.lr

    def step(self, state, env):
        '''
        Executes a step and updates the two agents

        Parameters:
            state (tuple): Sampled state
            env (OpenAI gym): Environment
        '''
        if not isinstance(state, tuple):
            state = (state, )
            
        best_action, _, support_best_action, selected = self.simple_action_selector(state)
        next_state, reward, done, _ = env.step(best_action)

        # Supporting agent update
        #state = (state, ) #TODO TMP FIX for 1d env!!
        self.support.update(state, next_state, best_action, reward)

        # MLGNG agent update
        self.mlgng_update_strategy(state, support_best_action)

        return next_state, reward, done, selected


    def train(self, env, num_episodes, stats, print_freq=50):
        print("#### Starting training #####")
        self.rewards = stats["cumulative_reward"]
        self.max_avg_reward = np.NINF

        self.init_adaptive_lr_params()
        for episode in range(num_episodes):
            self.episode = episode
            done = False
            step = 0
            cumulative_reward = 0
            selector_sequence = []

            start = time.time()
            state = env.reset()

            while not done:
                next_state, reward, done, selected = self.step(state, env)
                state = next_state
                
                cumulative_reward += reward
                selector_sequence.append(selected)
                step+=1

            stats["selector"][episode] = sum(selector_sequence)/len(selector_sequence)
            stats["cumulative_reward"][episode] = cumulative_reward
            stats["step"][episode] = step 
            stats["global_error"][episode] = self.mlgng.get_last_stat_tuple("global_error")
            stats["best_actions"].append(self.get_best_actions())
            stats["mlgng_nodes"].append(self.mlgng.get_nodes())
            stats["nodes"][episode] = self.mlgng.get_last_stat_tuple("vertices")
            stats["rate"][episode] = np.abs(self.lr)

            self.update_lr(episode)
            self.decay_param("discount", episode, decay_rate=0.015)
            self.support.epsilon = self.discount

            if self.episode >= 10:
                lower_window = self.episode-10
            else:
                lower_window = 0
            
            mean_reward = self.rewards[lower_window:self.episode].mean()
            if mean_reward > self.max_avg_reward:
                self.max_avg_reward = mean_reward
            
            stats["max_avg_reward"][episode] = self.max_avg_reward

            if (episode+1) % print_freq == 0:
                end = time.time() - start
                print("Episode {}/{}, Average Max Reward: {:.2f}, Global Error: {:.2f}, Total steps {}, Discount: {:.2f}, Time {:.3f}".format(
                    episode+1, 
                    num_episodes, 
                    stats["cumulative_reward"][episode-print_freq+1:episode].mean(),
                    stats["global_error"][episode].sum(),
                    stats["step"][episode], 
                    self.lr, 
                    end))
                self.mlgng.print_stats(one_line=True)

    def get_best_actions(self):
        length = np.prod(self.state_size)
        best_actions = np.empty((length, len(self.state_size)+1))

        for idx in range(length):
            state = np.unravel_index(idx, self.state_size)
            best_a, _, _, _ = self.simple_action_selector(state)
            best_actions[idx] = state + (best_a, )
        return best_actions.T