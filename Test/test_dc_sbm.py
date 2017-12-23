import networkx as nx
from dc_sbm import get_parameters, generate


def test_get_parameters():
    with open("toy.edges", "r") as net:
        G = nx.read_weighted_edgelist(net)
        parm = get_parameters(G)
        assert(sorted(parm[0].values()) == sorted([2/7., 2/7., 3/7.,
                                                   3/5., 1/5., 1/5.]))
        assert(sorted(parm[1].values()) == sorted([3, 1, 2]))
        assert(len(parm[2][0]) == 3)
        assert(len(parm[2][1]) == 3)


def test_generate():
    with open("toy.edges", "r") as net:
        G = nx.read_weighted_edgelist(net)
        H = generate(G)
        assert(len(G.nodes()) == len(H.nodes()))
