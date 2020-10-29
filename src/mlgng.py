from .gngu import GrowingNeuralGas
import numpy as np
from collections import defaultdict

class MultiLayerGrowingNeuralGas():
    '''
    Multi-Layer Growing Neural Gas
    from Gueriau et al, 2019

    Attributes:
        m (int): Number of layers
        ndim (int): State space dimensionality
        layers (dict): Dictionary of GrowingNeuralGas classes

    @author: Nassim Habbash
    '''

    def __init__(self, m, ndim):
        self.m = m
        self.layers = defaultdict()
        for i in range(m):
            self.layers[i] = GrowingNeuralGas(ndim=ndim)

    def __getitem__(self, key):
        return self.layers[key]

    def __setitem__(self, key, value):
        self.layers[key] = value
    
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

    def _format_state(self, s):
        
        if not isinstance(s, np.ndarray):
            s = np.ravel(np.array([s]))

        return s

    def update(self, s, m):
        '''
        Updates the m-th layer with the state s

        Parameters:
            s (tuple): Sampled state
            m (int): Layer
        '''
        s = self._format_state(s)
        self.layers[m].fit(s)

    def policy(self, s):
        '''
        Returns the top_k best actions given a state s.
        The best action is given by the GNG layer containing the closest node to s.

        Parameters:
            s (tuple): Sampled state
        '''
        s = self._format_state(s)

        A = np.empty(self.m)
        for i in range(self.m):
            _, _, error_w, _ = self.layers[i]._nearest_neighbors(s)
            A[i] = np.inf if error_w == None else error_w

        return A

    def stats(self):
        for i in range(self.m):
            print("> Layer: ", i)
            self.layers[i].stats()