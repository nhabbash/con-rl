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
        self.ndim = ndim
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
                self[i].set_parameters(**params)
        else:
            self[m].set_parameters(**params)

    def update_discount_rate(self, discount_rate, m=-1):
        if m == -1:
            for i in range(self.m):
                self[i].discount_rate = discount_rate
        else:
                self[m].discount_rate = discount_rate
                
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
        self[m].fit(s)

    def policy(self, s):
        '''
        Returns the best actions given a state s.
        The best action is given by the GNG layer containing the closest node to s.

        Parameters:
            s (tuple): Sampled state
        '''
        s = self._format_state(s)

        A = np.empty(self.m)
        for i in range(self.m):
            _, _, error_w, _ = self[i]._nearest_neighbors(s)
            A[i] = error_w
        
        best_action = np.argmin(A)
        if A[best_action] == np.inf:
            best_action = None

        return best_action

    def print_stats(self, one_line=False):
        if one_line:
            nodes = []
            edges = []
            for i in range(self.m):
                nodes.append(str(len(self[i].g.get_vertices())))
                edges.append(str(len(self[i].g.get_edges())))
            
            print("\t MLGNG nodes per action layer: "+" ".join(nodes))
        else:
            for i in range(self.m):
                print("> Layer: ", i)
                self[i].stats()

    def get_nodes(self):
        # Useful for data visualization
        fdata = np.array([[], [], []])

        for i in range(self.m):
            pos = self[i].g.vp.pos.get_2d_array(pos=np.arange(self.ndim))
            data = np.ones((pos.shape[0]+1, pos.shape[1]))*i # Add a column
            data[:-1,:] = pos
            fdata = np.hstack((fdata, data))

        return fdata

    def get_last_stat_tuple(self, name):
        # Gets last elements of the stats for logging
        dict = {}
        for action in range(self.m):
            stat = getattr(self[action].stats, name)
            val = 0
            if len(stat) > 0:
                val =  stat[-1]
            dict[action] = val

        return dict