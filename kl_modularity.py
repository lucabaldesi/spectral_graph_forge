#!/usr/bin/env python

import networkx as nx
import graph_utils as gu
import math
import itertools
import community


def random_graph_partition(G, C=2):
    # C is the number of communities
    part = {}
    for n in G.nodes():
        group = random.sample(range(C), 1)[0]
        part[n] = group
    return part


def node_bipartition(g1, g2):
    part = {}
    for node in g1:
        part[node] = 0
    for node in g2:
        part[node] = 1


def maximum_modularity(G):
    nodes = set(G.nodes())
    n = len(nodes)
    best_m = None

    for j in range(int(math.floor(float(n)/2))):
        print "J: " + str(j)
        for g in itertools.combinations(nodes, j):
            g1 = set(g)
            g2 = nodes.difference(g1)
            part = node_bipartition(g1, g2)
            m = gu.norm_modularity(G, part)
            if not best_m or m > best_m:
                best_m = m
        print "Maximum modularity: " + str(best_m)
    return best_m


def number_subgroups(n):
    s = 0
    for k in range(0, int(math.floor(float(n)/2)) + 1):
        s += math.factorial(n)/(math.factorial(k)*math.factorial(n-k))
    return s


if __name__ == "__main__":
    # G = nx.karate_club_graph()
    # m = maximum_modularity(G)
    # print community.modularity(community.best_partition(G), G)
    print "{0:.2e}".format(number_subgroups(3))
    print "{0:.2e}".format(number_subgroups(16))
    print "{0:.2e}".format(number_subgroups(32))
    print "{0:.2e}".format(number_subgroups(64))
    print "{0:.2e}".format(number_subgroups(128))
