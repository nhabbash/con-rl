from .mlgng import MultiLayerGrowingNeuralGas
from .qlearning import QLearningAgent
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

    def __init__(self, action_size, state_size, update_threshold = 20):
        self.action_size = action_size
        self.state_size = state_size

        shape = state_size + (self.action_size, )
        self.action_counter = np.zeros(shape=shape)
        self.update_threshold = update_threshold

    def init_support(self, support):
        self.support = support

    def init_mlgng(self, **params):
        self.mlgng = MultiLayerGrowingNeuralGas(m=self.action_size, ndim=params["ndim"])
        self.mlgng.set_layers_parameters(params, m=-1)

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
        # If the counter for an incoming (state, action) pair is 0, reset the counter for all state's actions to start a new consecutive count
        if self.action_counter[state][support_best_action] == 0:
            self.action_counter[state] = 0
        
        self.action_counter[state][support_best_action] += 1

        # MLGNG Agent update
        if self.action_counter[state][support_best_action] >= self.update_threshold:
            support_best_action = self.support.policy(state, exploitation=True)
            self.mlgng.update(state, support_best_action)
            self.mlgng.update_discount_rate(self.support.epsilon) # TODO remove dependance from support
            self.action_counter[state] = 0

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


    def train(self, env, num_episodes, stats):
        
        for episode in range(num_episodes):
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

            self.support.decay_param("epsilon")

            stats["selector"][episode] = sum(selector_sequence)/len(selector_sequence)
            stats["cumulative_reward"][episode] = cumulative_reward
            stats["step"][episode] = step 
            stats["global_error"][episode] = self.mlgng.get_last_stat_tuple("global_error")
            stats["best_actions"].append(self.get_best_actions())
            stats["mlgng_nodes"].append(self.mlgng.get_nodes())

                
            end = time.time() - start
            if (episode+1) % 50 == 0:
                print("Episode {}/{}, Average Max Reward: {}, Global Error: {:.2f}, Total steps {}, Epsilon: {:.2f}, Alpha: {:.2f}, Time {:.3f}".format(
                    episode+1, 
                    num_episodes, 
                    stats["cumulative_reward"][episode-10:episode].mean(),
                    stats["global_error"][episode].sum(),
                    stats["step"][episode], 
                    self.support.epsilon, 
                    self.support.alpha, 
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