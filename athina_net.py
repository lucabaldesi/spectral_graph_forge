import networkx as nx
import random


def bernoulli(p):
    if random.random() <= p:
        return 1
    else:
        return 0


def edge_probability(G):
    n = float(len(G.nodes()))
    m = float(len(G.edges()))
    p = m / (n * n / 2 - n)
    return p


def actual_modularity(G, c=2):
    degree = G.degree()
    m = 0
    n = len(G.nodes())
    L = float(len(G.edges()))
    for e in G.edges():
        c1 = e[0]/(n/c)
        c2 = e[1]/(n/c)
        if c1 == c2:
            m += 1 - float(degree[e[0]]*degree[e[1]])/L
    return m/L


def athina_net(pp, p=0.14338, c=2, n=34):
    '''
    Build a community graph
    n: total number of nodes
    c: number of communities
    p: intra community edge probability
    pp: inter community edge probability
    '''
    edges = []
    for k in range(c):
        edges += [(i+(n/c)*k, j+(n/c)*k)
                  for (i, j) in nx.erdos_renyi_graph(n/c, p).edges()]
    for k1 in range(c-1):
        for k2 in range(k1+1, c):
            for i in range(n/c):
                for j in range(n/c):
                    if bernoulli(pp) == 1:
                        edges.append((i+(n/c)*k1, j+(n/c)*k2))
    G = nx.Graph()
    G.add_edges_from(edges)
    return G
