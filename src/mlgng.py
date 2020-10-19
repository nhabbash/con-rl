from gngu import GrowingNeuralGas
import numpy as np
from collections import defaultdict

class MultiLayerGrowingNeuralGas():
    '''
    Multi-Layer Growing Neural Gas
    from Gueriau et al, 2019

    Attributes:
        m (int): Number of layers
        layers (dict): Dictionary of GrowingNeuralGas classes

    @author: Nassim Habbash
    '''

    def __init__(self, m):
        self.m = m
        self.layers = defaultdict()
        for i in range(m):
            self.layers[i] = GrowingNeuralGas()
    
    def set_layers_parameters(self, params, m=-1):
        '''
        Set the parameters for one GNG layer or for all

        Parameters:
            m (int): Layer to set, if -1 set to all
            params (dict): Dictionary of parameters to send to GNG
        '''

        if m == -1:
            for i in range(self.m):
                self.layers[i].set_parameters(**params)
        else:
            self.layers[m].set_parameters(**params)

    def update(self, s, m):
        '''
        Updates the m-th layer with the state s

        Parameters:
            s (tuple): Sampled state
            m (int): Layer
        '''
        self.layers[m].fit(s)

    def policy(self, s):
        '''
        Returns the best action given a state s.
        The best action is given by the GNG layer containing the closest node to s.

        Parameters:
            s (tuple): Sampled state
        '''
        
        best_action = None
        distance = np.inf

        for i in range(self.m):
            _, _, error_w, _ = self.layers[i]._nearest_neighbors(s)
            if distance >= error_w:
                distance = error_w
                best_action = i
        
        return best_action