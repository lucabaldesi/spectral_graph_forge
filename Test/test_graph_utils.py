import networkx as nx
import graph_utils as gu
import numpy as nm


def test_get_joint_degrees():
    G = nx.read_weighted_edgelist("toy.edges")
    A = nx.to_numpy_matrix(G)
    d = gu.get_joint_degrees(A)

    o = {1: {3: 2}, 2: {2: 2, 3: 2}, 3: {1: 2, 2: 2, 3: 2}}

    assert(d == o)


def test_avg_neighbour_degree():
    G = nx.read_weighted_edgelist("toy.edges")
    A = nx.to_numpy_matrix(G)

    annd = gu.get_avg_neighbour_degree(A)
    o = {1: 1, 2: 2, 3: 3}

    assert(annd == o)


def test_common_neigh_dist():
    G = nx.read_weighted_edgelist("toy.edges")
    A = nx.to_numpy_matrix(G)

    cnd = gu.get_common_neigh_dist(A)
    o = {1: 0.2}

    assert(cnd == o)


def test_kcoreness():
    G = nx.read_weighted_edgelist("toy.edges")
    A = nx.to_numpy_matrix(G)

    kcore = gu.get_kcoreness(A)
    o = {1: 6, 2: 3, 3: 0}

    assert(kcore == o)


def test_get_degree_betweeness():
    G = nx.read_weighted_edgelist("toy.edges")
    A = nx.to_numpy_matrix(G)

    dbet = gu.get_degree_betweeness(A)
    o = {1: 0, 2: 0, 3: (0.6 + 0.7)/2}

    assert(len(dbet) == len(o))
    assert(abs(dbet[1] - o[1]) < 0.0001)
    assert(abs(dbet[2] - o[2]) < 0.0001)
    assert(abs(dbet[3] - o[3]) < 0.0001)


def test_node_distance_dist():
    G = nx.read_weighted_edgelist("toy.edges")
    A = nx.to_numpy_matrix(G)

    dist = gu.node_distance_dist(A)
    o = {1: 6./15, 2: 5./15, 3: 4./15}

    assert(len(dist) == len(o))
    assert(abs(dist[1] - o[1]) < 0.0001)
    assert(abs(dist[2] - o[2]) < 0.0001)
    assert(abs(dist[3] - o[3]) < 0.0001)


def test_correlation():
    gu.correlation([1, 2, 3], [2, 9])


def test_correlate_dist_dict():
    d1 = {1: 2, 3: 45, 9: 2}
    d2 = {3: 2, 8: 4, 12: 45.3}
    v1 = nm.matrix("2. 0 45 0 0 0 0 0 2 0 0 0")
    v2 = nm.matrix("0 0 2 0 0 0 0 4 0 0 0 45.3")

    assert(gu.correlate_dist_dict(d1, d2) == gu.correlation(v1, v2))

    d1 = {1: 2, 3: 4, 4: 5, 0: 1}
    d2 = {3: 4, 0: 1, 4: 5, 1: 2}

    assert(gu.correlate_dist_dict(d1, d2) == 1)


def test_average_clustering():
    G = nx.read_weighted_edgelist("toy.edges")
    A = nx.to_numpy_matrix(G)

    assert(gu.average_clustering(A) < 7.0/18 + 0.001)
    assert(gu.average_clustering(A) > 7.0/18 - 0.001)
    assert(gu.average_clustering(A) < gu.average_clustering_old(A) + 0.001)
    assert(gu.average_clustering(A) > gu.average_clustering_old(A) - 0.001)


def test_is_connected():
    G = nx.read_weighted_edgelist("toy.edges")
    A = nx.to_numpy_matrix(G)

    assert(gu.is_connected(A))
    assert(gu.is_connected(nm.matrix("1 1 1; 1 1 1; 1 1 1")))
    assert(not gu.is_connected(nm.matrix("0 0 0; 0 0 0; 0 0 0")))
