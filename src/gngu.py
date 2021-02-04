from graph_tool import Graph
import numpy as np
from collections import namedtuple

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
        discount_rate (float): Rate by which discounts are applied
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

    def __init__(self, 
                ndim, 
                id=0, 
                discount_rate=1, 
                e_w=0.5, 
                e_n=0.1, 
                l=10, 
                a=0.5, 
                b=0.05, 
                k=1000.0, 
                max_nodes=100, 
                max_age=200, 
                node_multiplier=10, 
                min_error=5):
        
        self.g = Graph(directed=False)
        self.id = id
        self.ndim = ndim

        self.e_w = e_w
        self.e_n = e_n
        self.l = l
        self.a = a 
        self.b = b
        self.k = k
        self.discount_rate = discount_rate
        self.max_nodes = max_nodes
        self.max_age = max_age
        self.node_multiplier = node_multiplier
        self.min_error = min_error
        self.i = 0
        self.initialized = False

        self.stats = {
                        "global_error": [],
                        "global_utility": [],
                        "vertices": [],
                        "edges": []
                    }

        # Property maps for graph and edge variables
        self.g.ep.age = self.g.new_edge_property("int")
        self.g.vp.utility = self.g.new_vertex_property("float")
        self.g.vp.error = self.g.new_vertex_property("float")
        self.g.vp.action = self.g.new_vertex_property("string")
        self.g.vp.pos = self.g.new_vertex_property("vector<double>")

        # graph-tools properties
        self.g.set_fast_edge_removal(fast=True)
    
    def set_parameters(self, 
                ndim, 
                discount_rate=1, 
                e_w=0.5,
                e_n=0.1, 
                l=10, 
                a=0.5, 
                b=0.05, 
                k=1000.0, 
                max_nodes=100, 
                max_age=200,
                node_multiplier=10, 
                min_error=5):
                
        self.e_w = e_w
        self.e_n = e_n
        self.l = l
        self.a = a 
        self.b = b
        self.k = k
        self.discount_rate = discount_rate
        self.max_nodes = max_nodes
        self.max_age = max_age
        self.ndim = ndim
        self.node_multiplier = node_multiplier
        self.min_error = min_error

    def update_discount_rate(self, discount_rate):
        self.discount_rate = discount_rate

    def _nearest_neighbors(self, s):
        '''
        Return the two closest nodes to s and updates error
        '''

        winner = second = None
        error_w = error_s = np.inf
        
        if self.initialized:
            all_pos = self.g.vp.pos.get_2d_array(np.arange(self.ndim))
            # Squared distance
            s_col = s.reshape(-1, 1)
            distances = np.sum((all_pos - s_col) ** 2, axis=0)
            winner, second, *_ = np.argpartition(distances, 1)

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
        self.g.vp.pos[winner] += self.e_w*self.discount_rate*(s-self.g.vp.pos[winner])
    
        # Adapt neighbors if they exist
        neighbors = self.g.get_all_neighbors(winner)

        if neighbors.size > 0:
            all_pos = self.g.vp.pos.get_2d_array(np.arange(self.ndim))

            # Move winner's neighbors
            s_col = s.reshape(-1, 1)
            all_pos[:, neighbors] += self.e_n*self.discount_rate*(s_col-all_pos[:, neighbors])
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

    def _prune_nodes_by_isolation(self):
        '''
        Prunes nodes not connected to any edge
        '''
        # Remove unconnected nodes
        degrees = self.g.get_total_degrees(self.g.get_vertices())

        nodes = np.argwhere(degrees==0).flatten()
        #self.g.remove_vertex(nodes, fast=True)
        self.g.remove_vertex(nodes)

    def _prune_nodes_by_utility(self):
        '''
        Prunes nodes with the smallest utility if the highest error in the graph and smallest utility ratio is above k
        '''
        nodes_props = self.g.get_vertices(vprops=[self.g.vp.utility, self.g.vp.error])

        # Remove lowest utility node if the condition is met
        highest_error_node = np.argmax(nodes_props[:, -1], axis=0)
        highest_error = nodes_props[highest_error_node, -1]
        lowest_utility_node = np.argmin(nodes_props[:, -2], axis=0)
        lowest_utility = nodes_props[lowest_utility_node, -2]

        if highest_error > self.k * lowest_utility and self.initialized:
            self.g.remove_vertex(lowest_utility_node)

        return highest_error_node

    def _add_node_interpolation(self, highest_error_node):
        '''
        Add node to underrepresented areas according to the lambda insertion rate
        '''

        if self.i % self.l == 0 and len(self.g.get_vertices()) != self.max_nodes:
            # try:
            neighbors = self.g.get_all_neighbors(highest_error_node, vprops=[self.g.vp.error])
            # except:
            #     print(highest_error_node)
            #     print(self.id)
            if neighbors.size > 0:
                # highest_error_neighbor = np.argmax(neighbors[:, -1], axis=0)
                highest_error_neighbor = neighbors[np.argmax(neighbors[:, -1], axis=0), 0].astype(int)
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
                self.g.vp.utility[highest_error_node] *= self.a
                self.g.vp.utility[highest_error_neighbor] *= self.a

    def _add_node_perturbed(self, highest_error_node):
        '''
        Add node to underrepresented areas according to the lambda insertion by cloning and perturbing the highest error node
        '''

        if self.i % self.l == 0 and len(self.g.get_vertices()) != self.max_nodes:
            self.g.vp.error[highest_error_node] *= self.a
            self.g.vp.utility[highest_error_node] *= self.a

            neighbors = self.g.get_all_neighbors(highest_error_node)

            v = self.g.add_vertex()
            self.g.vp.error[v] = self.g.vp.error[highest_error_node]
            self.g.vp.utility[v] = self.g.vp.utility[highest_error_node]
            self.g.vp.pos[v] = self.g.vp.pos[highest_error_node] + np.random.normal(0, 0.3, self.ndim)

            edge_list = np.full((neighbors.shape[0], 2), int(v))
            edge_list[:, -1] = neighbors
            self.g.add_edge_list(edge_list)
            
    def _add_node_transfer(self, s):
        '''
        Add node
        '''
        if self.i % self.l == 0 and len(self.g.get_vertices()) != self.max_nodes:
            s = np.array(s) + np.random.normal(0, 0.1, self.ndim)
            v = self.g.add_vertex()

            winner, second, _, _ = self._nearest_neighbors(s)
            self.g.vp.error[v] = (self.g.vp.error[winner]+self.g.vp.error[second])/2
            self.g.vp.utility[v] = (self.g.vp.utility[winner]+self.g.vp.utility[second])/2
            self.g.vp.pos[v] = s

            edge_list = np.full((2, 2), int(v))
            edge_list[:, -1] = np.array([winner, second])
            self.g.add_edge_list(edge_list)
            

    def _discount(self):
        '''
        Discounts error and utility
        '''
        self.g.vp.error.a *= self.b
        self.g.vp.utility.a *= self.b

    def _add_init_node(self, s):
        '''
        Adds the inital node
        '''
        v = self.g.add_vertex()
        self.g.vp.error[v] = 0.0
        self.g.vp.utility[v] = 0.0
        self.g.vp.pos[v] = s

    def early_stopping(self):
        global_error = np.mean(self.g.vp.error.get_array())
        num_nodes = len(self.g.get_vertices())

        error_condition = global_error <= num_nodes*self.node_multiplier
        zero_condition = global_error > self.min_error

        return error_condition and zero_condition
    
    def fit(self, s, debug=False):
        if len(self.g.get_vertices()) < 2:
            self._add_init_node(s)
        else:
            winner, second = self._update_winner(s)
            self._adapt_neighborhood(winner, second, s)
            
            if not self.early_stopping():
                self._prune_edges()
                self._prune_nodes_by_isolation()
                highest_error_node = self._prune_nodes_by_utility()
                self._add_node_interpolation(highest_error_node)
                self.i+=1
            self._discount()

        self.initialized = False if len(self.g.get_vertices()) < 2 else True

        # Stats
        self.stats["global_error"].append(np.mean(self.g.vp.error.get_array()))
        self.stats["global_utility"].append(np.mean(self.g.vp.utility.get_array()))
        self.stats["vertices"].append(len(self.g.get_vertices()))
        self.stats["edges"].append(len(self.g.get_edges()))
        if debug:
            #TODO: send data to graph here
            self.print_stats()

    def print_stats(self, full=False):
        if self.i%100!=0 and full:
            return

        print("Iterations: ", self.i)
        print("Graph properties: ")
        print("\t Order: {}".format(self.stats["vertices"][self.i]))
        print("\t Size: {}".format(self.stats["edges"][self.i]))
        print("\t Global error: {}".format(self.stats["global_error"][self.i]))