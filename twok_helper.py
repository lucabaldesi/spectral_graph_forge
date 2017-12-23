import sys
import networkx as nx

sys.path.append("twofiveK")
from Estimation import Estimation
from Generation import Generation


def gen_25k_graph(O, names=None):
    error = 0.05

    G = nx.Graph(O)
    test25 = Estimation()
    gen25 = Generation()
    test25.load_graph("", graph=G)
    test25.calcfull_CCK()
    test25.calcfull_JDD()
    gen25.set_JDD(test25.get_JDD('full'))
    gen25.set_KTRI(test25.get_KTRI('full'))
    gen25.construct_triangles_2K()
    gen25.mcmc_improved_2_5_K(error_threshold=error)

    H = gen25.G
    for i in range(len(O.nodes())-len(H.nodes())):
        H.add_node(len(H.nodes()))
    if names:
        dknames = H.nodes()
        mapping = {dknames[i]: names[i] for i in range(len(names))}
        H = nx.relabel_nodes(H, mapping)

    assert(len(H.nodes()) == len(O.nodes()))
    return H
