from .mlgng import MultiLayerGrowingNeuralGas
from .qlearning import QLearning
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

    def __init__(self, num_actions, ndim, update_threshold = 20):
        self.num_actions = num_actions
        self.support = QLearning(num_actions=num_actions)
        self.mlgng = MultiLayerGrowingNeuralGas(m=num_actions, ndim=ndim)
        self.action_counter = defaultdict(lambda: np.zeros(num_actions))
        self.update_threshold = update_threshold

    def _simple_action_selector(self, state):
        '''
        Action selection, if MLGNG has an action, select it, otherwise get it from QL

        Parameters:
        state (tuple): Sampled state
        '''

        # QL action - choose the action with the highest Q value with epsilon probability to explore
        action_probs = self.support.policy(state)
        q_best_action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

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
            selected = "Support"
        else: 
            best_action = mlgng_best_action
            selected = "MLGNG"

        return best_action, mlgng_best_action, q_best_action, selected

    def mlgng_update_strategy(self, state, support_best_action):
        '''
        MLGNG is updated only after a state reaches a threshold of consecutive actions from the supporting agent
        '''

        # Consecutive action counter update
        # If the counter for an incoming (state, action) pair is 0, reset the counter for all state's actions to start a new consecutive count
        if self.action_counter[state][support_best_action] == 0:
            self.action_counter[state] = np.zeros(self.num_actions)
        
        self.action_counter[state][support_best_action] += 1

        # MLGNG Agent update
        if self.action_counter[state][support_best_action] >= self.update_threshold:
            self.mlgng.update(state, support_best_action)

    def step(self, state, env):
        '''
        Executes a step and updates the two agents

        Parameters:
            state (tuple): Sampled state
            env (OpenAI gym): Environment
        '''

        # TODO: discretization of s

        best_action, _, support_best_action, selected = self._simple_action_selector(state)
        next_state, reward, done, _ = env.step(best_action)

        # Supporting agent update
        self.support.update(state, next_state, best_action, reward)

        # MLGNG agent update
        self.mlgng_update_strategy(state, support_best_action)
        # print(next_state, best_action, selected)
        return next_state, reward, done, selected