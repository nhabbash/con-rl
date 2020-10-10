from graph_tool.all import *

class GNGU():
'''
Growing Neural Gas with Utility implementation 
from Fritzke, 1997
'''

def __init__(self, e_w, e_n, _lambda, max_nodes, max_age):
    self.g = Graph(directed=False)

    # Fraction of distance to move the winner node and its neighboring nodes towards the current signal
    self.e_w = e_w
    self.e_n = e_n

    # Number of iterations to insert a new node (fixed insertion rate)
    # TODO: error-based insertion rate (ie when mean squared error is larger than a constant add node)
    self._lambda = _lambda

    # Error decrease parameter for nodes with largest and neighboring largest errors after insertion
    self.alpha = alpha 

    # Global error decrease parameter
    self.beta = beta

    # Utility removal sensitiveness, lower = frequent deletions, k has to be a little higher than the mean U/error ratio
    self.k = k

    self.max_nodes = max_nodes
    self.max_age = max_age

    # Property maps for graph and edge variables
    g.ep.edge_age = g.new_edge_property("int")
    g.vp.node_utility = g.new_vertex_property("float")
    g.vp.node_error = g.new_vertex_property("float")
    g.vp.node_action = g.new_vertex_property("string")
    g.vp.pos = g.new_vertex_property("vector<double>")
