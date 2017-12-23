import networkx as nx
import sys

sys.path.append("modular_graph_generator")
import random_modular_generator_variable_modules as mgg
import sequence_generator as sg
import community


def get_dists(degdist, moddist):
    if degdist == "regular":
        degfunction = sg.regular_sequence
    elif degdist == "poisson":
        degfunction = sg.poisson_sequence
    elif degdist == "scalefree":
        degfunction = sg.scalefree_sequence
    elif degdist == "geometric":
        degfunction = sg.geometric_sequence
    else:
        print "ERROR"
    if moddist == "regular":
        modfunction = sg.regular_sequence
    elif moddist == "poisson":
        modfunction = sg.poisson_sequence
    elif moddist == "scalefree":
        modfunction = sg.scalefree_sequence
    elif moddist == "geometric":
        modfunction = sg.geometric_sequence
    else:
        print "ERROR"

    return degfunction, modfunction


def random_graph(G, degdist="regular", moddist="regular"):
    degfunction, modfunction = get_dists(degdist, moddist)
    N = len(G.nodes())
    part = community.best_partition(G)
    Q = community.modularity(part, G)
    m = len(set(part.values()))
    d = nx.degree(G)
    d = sum(d.values())/float(len(d))
    d = int(round(d))
    return mgg.generate_modular_networks(N, degfunction, modfunction,
                                         Q, m, d)


if __name__ == "__main__":
    G = nx.karate_club_graph()
    random_graph(G, degdist="scalefree")
