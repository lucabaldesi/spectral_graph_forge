import networkx as nx
import numpy as nm

from eigen_graph import eigen_centrality, vec_dist, build_matrix, \
    sample_simm_matrix, sample_simm_matrix2, modularity_clone_matrix, \
    laplacian_clone_matrix, generate_matrix, synthetic_modularity_matrix, \
    leading_eigenvector


def test_eigen_centrality():
    with open("toy.edges", "r") as net:
        G = nx.read_weighted_edgelist(net)
        A = nx.to_numpy_matrix(G)
        x, l = eigen_centrality(A)

        y = nm.matrix("0.200305; 0.200305; 0.256073; \
                      0.182829; 0.080244; 0.080244")

        assert(vec_dist(x, y) < 0.001)
        assert(abs(l - 2.2784) < 0.001)

    A = nm.matrix("-0.333333   0.500000   0.666667  -0.166667  -0.500000 \
                   -0.166667;   0.500000  -0.750000   0.500000  -0.250000 \
                   0.250000  -0.250000;  0.666667   0.500000  -0.333333 \
                   -0.166667  -0.500000  -0.166667; -0.166667  -0.250000 \
                   -0.166667  -0.083333   0.750000  -0.083333; -0.500000 \
                   0.250000  -0.500000   0.750000  -0.750000   0.750000; \
                   -0.166667  -0.250000  -0.166667  -0.083333   0.750000 \
                   -0.083333")
    x, l = leading_eigenvector(A)
    y = nm.matrix("0.21001; -0.46692;  0.21001; -0.32423;  0.69536; -0.32423")
    assert(vec_dist(x, y) < 0.001)
    assert(abs(l + 1.9193) < 0.001)


def test_build_matrix():
    G = nx.read_weighted_edgelist("toy.edges")
    A = nx.to_numpy_matrix(G)
    x, l = eigen_centrality(A)

    B = build_matrix(x, l, 0.02)
    y, m = eigen_centrality(B, maxiter=10000)
    assert(vec_dist(x, y) < 0.13)
    assert((l / m) - 1 < 0.1)


def test_generate_matrix():
    G = nx.read_weighted_edgelist("toy.edges")
    A = nx.to_numpy_matrix(G)
    x, l = eigen_centrality(A)

    B = generate_matrix(x, l*x, 0.02, nm.ravel(A * nm.ones((6, 1))))
    y, m = eigen_centrality(B, maxiter=10000)
    assert(vec_dist(x, y) < 0.13)
    assert((l / m) - 1 < 0.1)


def test_sample_matrix():
    G = nx.read_weighted_edgelist("toy.edges")
    A = nx.to_numpy_matrix(G)
    B = sample_simm_matrix(A, 6)

    assert(nm.array_equal(A, B))


def test_sample_matrix2():
    G = nx.read_weighted_edgelist("toy.edges")
    A = nx.to_numpy_matrix(G)
    B = sample_simm_matrix2(A, 6)

    assert(nm.array_equal(A, B))


def test_modularity_clone_matrix():
    G = nx.read_weighted_edgelist("toy.edges")
    A = nx.to_numpy_matrix(G)
    B = modularity_clone_matrix(A, 6)

    assert(nm.array_equal(A, B))


def test_laplacian_clone_matrix():
    G = nx.read_weighted_edgelist("toy.edges")
    A = nx.to_numpy_matrix(G)
    B = laplacian_clone_matrix(A, 6)

    assert(nm.array_equal(A, B))


def test_synthetic_modularity_matrix():
    G = nx.read_weighted_edgelist("toy.edges")
    A = nx.to_numpy_matrix(G)
    K = nm.ones((1, 6)) * A
    x, l = leading_eigenvector(A - nm.transpose(K) * K /
                               float(sum(nm.ravel(K))))

    B = synthetic_modularity_matrix(A, 0.001)
    K = nm.ones((1, 6)) * B
    y, m = leading_eigenvector(B - nm.transpose(K) * K /
                               float(sum(nm.ravel(K))))
    assert(vec_dist(x, y) < 0.1)
    assert((l / m) - 1 < 0.1)
