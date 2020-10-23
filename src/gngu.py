import graph_tool.all as gt
import numpy as np

class GrowingNeuralGas():
    '''
    Growing Neural Gas with Utility implementation 
    from Fritzke, 1997

    Steps:
        1. Update: update winner node properties and discount errors
        2. Adapt
        3. Refine
        4. Add

    Attributes:
        ndim (int): State space dimensionality
        e_w (float): Fraction of distance to move the winner node towards the current signal
        e_n (float): Fraction of distance to move the neighbors of the winner node towards the current signal
        i (int): Current number of iterations
        l (int): Number of iterations for new node insertions (fixed insertion rate)
        a (float): Local error discount parameter for nodes with largest and neighboring largest errors after insertion
        b (float): Global error discount parameter
        k (float): Utility removal sensitiveness, lower = frequent deletions, for stability k has to be a little higher than the mean U/error ratio
        max_nodes (int): Maximum number of nodes allowed in the GNG
        max_age (int): Maximum age an edge can reach before removal
        initialized (bool): is True when GNG has at least two nodes 

    @author: Nassim Habbash
    '''

    def __init__(self, ndim, e_w=0.5, e_n=0.1, l=10, a=0.5, b=0.05, k=1000.0, max_nodes=100, max_age=200):
        
        self.g = gt.Graph(directed=False)
        self.ndim = ndim
        self.e_w = e_w
        self.e_n = e_n
        self.l = l
        self.a = a 
        self.b = b
        self.k = k
        self.max_nodes = max_nodes
        self.max_age = max_age
        self.i = 0
        self.initialized = False

        # Property maps for graph and edge variables
        self.g.ep.age = self.g.new_edge_property("int")
        self.g.vp.utility = self.g.new_vertex_property("float")
        self.g.vp.error = self.g.new_vertex_property("float")
        self.g.vp.action = self.g.new_vertex_property("string")
        self.g.vp.pos = self.g.new_vertex_property("vector<double>")

        # graph-tools properties
        self.g.set_fast_edge_removal(fast=True)
    
    def set_parameters(self, ndim, e_w=0.5, e_n=0.1, l=10, a=0.5, b=0.05, k=1000.0, max_nodes=100, max_age=200):
        self.e_w = e_w
        self.e_n = e_n
        self.l = l
        self.a = a 
        self.b = b
        self.k = k
        self.max_nodes = max_nodes
        self.max_age = max_age
        self.ndim = ndim

    def _nearest_neighbors(self, s):
        '''
        Return the two closest nodes to s and updates error
        '''

        winner = second = error_w = error_s = None
        if len(self.g.get_vertices()) >= 2:
            all_pos = self.g.vp.pos.get_2d_array(np.arange(self.ndim))
            # Squared distance
            s_col = s.reshape(-1, 1)
            distances = np.sum((all_pos - s_col) ** 2, axis=0)
            # print(all_pos)
            # print(s_col)
            # print(distances)
            winner, second, *_ = np.argpartition(distances, 1) 
            # TODO: check if kDTree search is faster than np.argpartition 

            error_w = distances[winner]
            error_s = distances[second]

        return winner, second, error_w, error_s

    def _update_winner(self, s):
        '''
        Update age, error and utility of the winner node.
        '''

        winner, second, error_w, error_s = self._nearest_neighbors(s)

        self.g.vp.error[winner] += error_w # Error update
        self.g.vp.utility[winner] += error_s - error_w # Utility update

        edges = self.g.get_all_edges(winner, eprops=[self.g.edge_index])
        self.g.ep.age.a[edges[:,-1]] += 1 # Increment age of the winner's edges
            
        return winner, second

    def _adapt_neighborhood(self, winner, second, s):
        '''
        Move the winner node and its topological neighbors towards s, add an edge between the two winner nodes
        '''
        # Adapt winner's position
        self.g.vp.pos[winner] += self.e_w*(s-self.g.vp.pos[winner])
    
        # Adapt neighbors if they exist
        neighbors = self.g.get_all_neighbors(winner)

        if neighbors.size > 0:
            all_pos = self.g.vp.pos.get_2d_array(np.arange(self.ndim))

            # Move winner's neighbors
            s_col = s.reshape(-1, 1)
            all_pos[:, neighbors] += self.e_n*(s_col-all_pos[:, neighbors])
            self.g.vp.pos.set_2d_array(all_pos)

        # Connect the winner nodes and resets the edge age to 0
        e = self.g.edge(winner, second)
        if not e:
            e = self.g.add_edge(winner, second)
        self.g.ep.age[e] = 0


    def _prune_edges(self):
        '''
        Prunes edges with age exceeding max_age
        '''

        all_edges = self.g.get_edges(eprops=[self.g.ep.age, self.g.edge_index])

        mask = all_edges[:, -2] >= self.max_age
        old_edges = all_edges[mask]

        for e in old_edges:
            edge = self.g.edge(e[0], e[1])
            self.g.remove_edge(edge)

    def _prune_nodes(self):
        '''
        Prunes nodes not connected to any edge and the node with the smallest utility if the highest error in the graph and smallest utility ratio is above k
        '''

        # Remove unconnected nodes
        degrees = self.g.get_total_degrees(self.g.get_vertices())

        nodes = np.argwhere(degrees==0).flatten()
        self.g.remove_vertex(nodes, fast=True)
        
        nodes_props = self.g.get_vertices(vprops=[self.g.vp.utility, self.g.vp.error])

        # Remove lowest utility node if the condition is met
        highest_error_node = np.argmax(nodes_props[:, -1], axis=0)
        highest_error = nodes_props[highest_error_node, -1]
        lowest_utility_node = np.argmin(nodes_props[:, -2], axis=0)
        lowest_utility = nodes_props[lowest_utility_node, -2]

        if highest_error > self.k * lowest_utility and len(self.g.get_vertices()) > 2:
            self.g.remove_vertex(lowest_utility_node, fast=True)

        return highest_error_node

    def _add_node(self, highest_error_node):
        '''
        Add node to underrepresented areas according to the lambda insertion rate
        '''
        # TODO: error-based insertion rate (ie when mean squared error is larger than a threshold add node)

        if self.i % self.l == 0 and len(self.g.get_vertices()) != self.max_nodes:
            # print(highest_error_node)
            neighbors = self.g.get_all_neighbors(highest_error_node, vprops=[self.g.vp.error])

            if neighbors.size > 0:
                highest_error_neighbor = np.argmax(neighbors[:, -1], axis=0)
                
                p1 = self.g.vp.pos[highest_error_node]
                p2 = self.g.vp.pos[highest_error_neighbor]

                u1 = self.g.vp.utility[highest_error_node]
                u2 = self.g.vp.utility[highest_error_neighbor]
                
                v = self.g.add_vertex()
                self.g.vp.error[v] = self.g.vp.error[highest_error_node]
                self.g.vp.utility[v] = (u1+u2)*0.5
                self.g.vp.pos[v] = (np.array(p1)+np.array(p2))*0.5

                self.g.add_edge_list([  [v, highest_error_node], 
                                        [v, highest_error_neighbor]])
                
                self.g.vp.error[highest_error_node] *= self.a
                self.g.vp.error[highest_error_neighbor] *= self.a
            
    def _discount(self):
        '''
        Discounts error and utility
        '''
        self.g.vp.error.a -= self.g.vp.error.a * self.b
        self.g.vp.utility.a -= self.g.vp.error.a * self.b

    def _add_init_node(self, s):
        '''
        Adds the inital node
        '''
        v = self.g.add_vertex()
        self.g.vp.error[v] = 0.0
        self.g.vp.utility[v] = 0.0
        self.g.vp.pos[v] = s
    
    def fit(self, s, debug=False):
        if len(self.g.get_vertices()) < 2:
            self._add_init_node(s)
        else:
            winner, second = self._update_winner(s)
            self._adapt_neighborhood(winner, second, s)
            self._prune_edges()
            highest_error_node = self._prune_nodes()
            self._add_node(highest_error_node)
            self._discount()

        self.i+=1
        self.initialized = False if len(self.g.get_vertices()) < 2 else True

        if debug:
            self.stats()

    def stats(self, slow=True):
        # TODO: Error tracking
        if self.i%100!=0 and slow:
            return

        print("Iteration: ", self.i)
        print("Graph properties: ")
        print(self.g)
    
    def n_clusters(self):
        pass
