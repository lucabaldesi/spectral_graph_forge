#!/usr/bin/env python
# This module is an implementation from the paper:
# Karrer, Brian, and Mark EJ Newman. "Stochastic blockmodels and community
# structure in networks." Physical Review E 83.1 (2011): 016107.

import networkx as nx
import numpy as nm
import bisect
import random
import community


def sample_discrete(dist):
    # sample a discrete distribution dist with values = dist.keys() and
    # probabilities = dist.values()

    i = 0
    acc = 0
    values = {}
    probs = []
    for e in dist:
        values[i] = e
        acc += dist[e]
        probs.append(acc)
        i += 1

    rand = random.random()
    pos = bisect.bisect(probs, rand)
    return values[pos]


def get_parameters(G):
    part = community.best_partition(G)
    M = {}
    for e in G.edges():
        r = part[e[0]]
        s = part[e[1]]
        el = tuple(sorted([r, s]))
        M[el] = M.get(el, 0) + 1

    g = {}
    for k, v in part.items():
        g[v] = g.get(v, []) + [k]

    k = G.degree()
    K = {}
    for c in g:
        K[c] = sum([k[i] for i in g[c]])

    t = k
    for e in t:
        if t[e] != 0:
            t[e] = float(t[e])/K[part[e]]

    return (t, M, g)


def generate_from_parameters(t, w, g):
    G = nx.Graph()
    for i in g:
        G.add_nodes_from(g[i])

    # generate num of edges
    M = w.copy()
    for c in M:
        M[c] = nm.random.poisson(M[c])

    # assign edges to vertices
    edges = []
    for c in M:
        r = c[0]
        s = c[1]
        for i in range(M[c]):
            n1 = sample_discrete({j: t[j] for j in g[r]})
            n2 = sample_discrete({j: t[j] for j in g[s]})
            edges.append((n1, n2))

    G.add_edges_from(edges)
    return G


def generate(G, names=None):
    t, w, g = get_parameters(G)
    return generate_from_parameters(t, w, g)


if __name__ == "__main__":
    G = nx.karate_club_graph()
    print G.nodes()
    t, w, g = get_parameters(G)
    H = generate_from_parameters(t, w, g)
    print H.nodes()
