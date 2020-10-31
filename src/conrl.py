from .mlgng import MultiLayerGrowingNeuralGas
from .qlearning import QLearningAgent
from collections import defaultdict

import numpy as np

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

    def __init__(self, action_size, ndim, state_grid, update_threshold = 20):
        self.action_size = action_size
        self.support = QLearningAgent(action_size=action_size, state_grid=state_grid)
        self.mlgng = MultiLayerGrowingNeuralGas(m=action_size, ndim=ndim)

        self.state_grid = state_grid
        self.state_size = tuple(len(dim) + 1 for dim in state_grid)
        shape = self.state_size + (self.action_size, )
        self.action_counter = np.zeros(shape=shape)
        self.update_threshold = update_threshold

    def _simple_action_selector(self, state):
        '''
        Action selection, if MLGNG has an action, select it, otherwise get it from QL

        Parameters:
        state (tuple): Sampled state
        '''

        # QL action - choose the action with the highest Q value with epsilon probability to explore
        q_best_action = self.support.policy(state)

        # MLGNG action - choose the action layer with the closest state
        action_distances = self.mlgng.policy(state)
        # If all action distances are np.inf then there is no initialized layer in MLGNG
        mlgng_best_action = None
        if not np.all(action_distances == np.inf):
            mlgng_best_action = np.argmin(action_distances)

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
            self.mlgng.update(state, support_best_action)

    def step(self, state, env, discretize=None):
        '''
        Executes a step and updates the two agents

        Parameters:
            state (tuple): Sampled state
            env (OpenAI gym): Environment
        '''

        if discretize:
            state = discretize(state, self.state_grid)

        best_action, _, support_best_action, selected = self._simple_action_selector(state)
        next_state, reward, done, _ = env.step(best_action)
        
        if discretize:
            next_state = discretize(next_state, self.state_grid)

        # Supporting agent update
        state = (state, ) #TODO TMP FIX!!
        self.support.update(state, next_state, best_action, reward)

        # MLGNG agent update
        self.mlgng_update_strategy(state, support_best_action)
        # print(next_state, best_action, selected)
        return next_state, reward, done, selected
