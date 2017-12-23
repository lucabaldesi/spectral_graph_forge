#!/usr/bin/env python

import numpy as nm
import random as rnd
import itertools as its
import networkx as nx
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys

import ssum
import graph_utils as gu


__eigen_debug = False
__eigen_info = False

scaling_fun = gu.truncate


def debug(s):
    if __eigen_debug:
        print "[DEBUG] " + str(s)
    return 0


def info(s):
    if __eigen_info:
        print "[INFO] " + str(s)
    return 0


def vec_dist(x, y):
    return (nm.linalg.norm(nm.sort(nm.ravel(x)) - nm.sort(nm.ravel(y))))


def leading_eigenvector(A, eps=0.000001, maxiter=float("inf")):
    n = A.shape[0]
    # m = 0.15  # Google magic number
    # S = nm.ones((n, n))
    # A = S * m + A * (1 - m)
    x0 = nm.random.random((n, 1))
    x0 = x0 / nm.linalg.norm(x0)

    i = 0
    x1 = A * x0
    l = nm.linalg.norm(x1)
    x1 = x1 / l
    while (nm.linalg.norm(x0-x1) > eps) and i < maxiter:
        x0 = x1
        x1 = A * x0
        l = float((nm.transpose(x1)*x0) / (nm.transpose(x0)*x0))
        x1 = x1 / l
        x1 /= nm.linalg.norm(x1)
        i += 1
        # print str(x1) + " " + str(l) + " " +\
        #     str(nm.linalg.norm(nm.ravel(x0)-nm.ravel(x1)))

    if x1[0] < 0:  # canonical (crappy) form
        x1 *= -1

    return x1, float(l)


def eigen_centrality(A, eps=0.000001, maxiter=float("inf")):
    x, l = leading_eigenvector(A, eps, maxiter)
    s = sum(x)
    x = x / s
    return x, l


def truncate_vector(x, precision):
    n = x.shape[0]
    j = 0
    while ((10**j) * precision) < 1:
        j += 1

    for i in range(n):
        x[i] = round(x[i], j)

    return x


def truncate_value(l, precision):
    j = 0
    while ((10**j) * precision) < 1:
        j += 1
    return round(l, j)


def good_solution(sol, x, s, goal, precision):
    tot = s
    debug("evaluating " + str(sol))
    for i in sol:
        tot += x.item(i)
    debug("tot: " + str(tot) + ", goal: " + str(goal))
    if abs(tot - goal) <= precision:
        return 1
    return 0


def select_neighbours_factorial(s, free, x, goal):
    not_found = 1
    i = 1

    while not_found and i <= len(free):
        for sol in its.combinations(free, i):
            if good_solution(sol, x, s, goal, 0.0001):
                not_found = 0
                break
        i += 1

    if not_found:
        return []
    else:
        debug("Found: " + str(sol))
        return sol


def select_neighbours(s, free, x, goal, precision, k):
    debug("Free: " + str(x[free, :]))
    debug("Goal: " + str(goal-s))
    debug("Neigh: " + str(k))
    sel = []
    if k >= 0:
        sel = ssum.gssum(x[free, :], goal - s, precision, k)
    debug(sel)
    sel = [free[int(v)] for v in sel]
    debug("Sum: " + str(sum(x[sel, :])))
    return sel


def build_matrix(x, l, precision):
    return generate_matrix(x, l*x, precision)


def generate_matrix(x, b, precision, degrees=None):
    '''
    Generate a random matrix A such that A*x = b up to a given precision.
    Both x and b must be numpy vectors.
    '''
    n = max(x.shape)
    nodes = range(n)
    A = nm.zeros((n, n))
    one = nm.ones(n)

    next_node_i = 1
    if degrees is None:
        nodes = nm.array(nodes)
    else:
        nodes = nm.argsort(degrees)
    for node in nodes:
        debug("node " + str(node))
        value = float(nm.dot(A[node, :], x))
        num_neighs = 0
        if degrees is not None:
            num_neighs = degrees[node] - nm.dot(A[node, :], one)
            if num_neighs == 0:
                num_neighs = -1

        debug("free " + str(nodes[next_node_i:]))
        neigh = nodes[next_node_i:].copy()
        rnd.shuffle(neigh)
        neigh = select_neighbours(value, neigh,
                                  x, float(b[node]), precision, num_neighs)
        for i in neigh:
            A[node, i] = 1
        A[:, node] = A[node, :]
        debug("neighbourhood " + str(A[node, :]))
        next_node_i += 1

    return nm.matrix(A)


def octave_print(A):
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            sys.stdout.write(str(A[i, j]) + " ")
        sys.stdout.write(";")
    print ""


def load_centrality():
    # A = nm.matrix("0 1 0 0 0 0; 1 0 1 0 0 0; 0 1 0 1 0 0; 0 0 1 0 1 1;\
    #               0 0 0 1 0 0; 0 0 0 1 0 0")
    with open("jazz.edges", "r") as net:
        G = nx.read_weighted_edgelist(net)
    fig = plt.figure()
    nx.draw(G)
    fig.savefig("given_graph.png")
    print "Average clustering:" + str(nx.average_clustering(G))
    # plt.show()

    info("Generating the matrix..")
    A = nx.to_numpy_matrix(G)
    info("Computing the centrality..")
    x, l = eigen_centrality(A)
    return x, l


def print_graph(G):
    plt.figure()
    nx.draw(G)
    plt.show()


def sample_simm_matrix(A, level, sorteigs=True):
    d, V = nm.linalg.eigh(A)
    n = len(d)
    if sorteigs:
        k = nm.argsort(nm.abs(d))
    else:
        k = range(n)
    B = nm.zeros(nm.shape(A))

    for i in range(1, level + 1):
        B += d[k[n-i]] * V[:, k[n-i]] * nm.transpose(V[:, k[n-i]])

    B -= nm.min(B)
    nm.fill_diagonal(B, nm.zeros((1, n)))
    B /= nm.max(B)

    for i in range(n-1):
        B[i, i+1:] = stats.bernoulli.rvs(B[i, i+1:],
                                         random_state=nm.random.RandomState())
        B[i+1:, i] = B[i, i+1:]

    return nm.matrix(B)


def mat_spect_approx(A, level, sorteigs=True, reverse=False, absolute=True):
    d, V = nm.linalg.eigh(A)
    d = nm.ravel(d)
    n = len(d)
    if sorteigs:
        if absolute:
            k = nm.argsort(nm.abs(d))
        else:
            k = nm.argsort(d)
        # ordered from the lowest to the highest
    else:
        k = range(n)
    if not reverse:
        k = nm.flipud(k)

    z = nm.zeros((n, 1))
    for i in range(level, n):
        V[:, k[i]] = z

    B = V*nm.diag(d)*nm.transpose(V)
    return B


def sample_simm_matrix2(A, level, sorteigs=True):
    if level == 0:
        level = 1
    n = min(A.shape)
    B = mat_spect_approx(A, level, sorteigs, absolute=True)

    nm.fill_diagonal(B, nm.zeros((1, n)))
    # B -= nm.min(B)
    # B /= nm.max(B)
    B = nm.vectorize(scaling_fun, otypes=[nm.float])(B)
    nm.fill_diagonal(B, nm.zeros((1, n)))

    for i in range(n-1):
        B[i, i+1:] = stats.bernoulli.rvs(B[i, i+1:],
                                         random_state=nm.random.RandomState())
        B[i+1:, i] = nm.transpose(B[i, i+1:])

    return nm.matrix(B)


def modularity_clone_matrix(A, level):
    n = A.shape[1]
    K = nm.ones((1, n)) * A
    B = A - nm.transpose(K) * K / float(sum(nm.ravel(K)))
    B = mat_spect_approx(B, level, sorteigs=True, absolute=True)

    B = B + nm.transpose(K) * K / float(sum(nm.ravel(K)))

    nm.fill_diagonal(B, nm.zeros((1, n)))
    # B -= nm.min(B)
    # B /= nm.max(B)
    B = nm.vectorize(scaling_fun, otypes=[nm.float])(B)
    nm.fill_diagonal(B, nm.zeros((1, n)))

    for i in range(n-1):
        B[i, i+1:] = stats.bernoulli.rvs(B[i, i+1:])  # ,
        # random_state=nm.random.RandomState())
        B[i+1:, i] = nm.transpose(B[i, i+1:])

    return nm.matrix(B)


def laplacian_clone_matrix(A, level):
    if level == 0:
        level = 1
    n = A.shape[1]
    K = nm.ravel(nm.ones((1, n)) * A)
    L = nm.diag(K) - A
    L = mat_spect_approx(L, level, reverse=True, absolute=True)

    nm.fill_diagonal(L, nm.zeros((1, n)))
    L = -L
    L -= nm.min(L)
    L /= nm.max(L)
    # L = nm.vectorize(gu.ilogit, otypes=[nm.float])(L)
    nm.fill_diagonal(L, nm.zeros((1, n)))

    for i in range(n-1):
        L[i, i+1:] = stats.bernoulli.rvs(L[i, i+1:],
                                         random_state=nm.random.RandomState())
        L[i+1:, i] = nm.transpose(L[i, i+1:])

    return nm.matrix(L)


def modspec_clone_matrix(A, level):
    if level == 0:
        level = 1
    n = A.shape[1]
    K = nm.ones((1, n)) * A
    B = A - nm.transpose(K) * K / float(sum(nm.ravel(K)))
    B = mat_spect_approx(B, level, absolute=False)

    B = (B + nm.transpose(K) * K / float(sum(nm.ravel(K))))
    B += mat_spect_approx(A, level, absolute=True)

    L = nm.diag(K) - A
    L = mat_spect_approx(L, level, reverse=False, absolute=False)
    B += nm.diag(K) - L

    nm.fill_diagonal(B, nm.zeros((1, n)))
    # B -= nm.min(B)
    # B /= nm.max(B)
    B = nm.vectorize(scaling_fun, otypes=[nm.float])(B)
    nm.fill_diagonal(B, nm.zeros((1, n)))

    for i in range(n-1):
        B[i, i+1:] = stats.bernoulli.rvs(B[i, i+1:],
                                         random_state=nm.random.RandomState())
        B[i+1:, i] = nm.transpose(B[i, i+1:])

    return nm.matrix(B)


def franky_clone_matrix(A, level):
    n = A.shape[1]
    K = nm.ones((1, n)) * A

    B = A - nm.transpose(K) * K / float(sum(nm.ravel(K)))
    B = mat_spect_approx(B, 1)
    B = (B + nm.transpose(K) * K / float(sum(nm.ravel(K))))

    L = nm.diag(K) - A
    L = mat_spect_approx(L, level, reverse=True)
    nm.fill_diagonal(L, nm.zeros((1, n)))
    L = -L

    B = B + L + mat_spect_approx(A, 1)

    nm.fill_diagonal(B, nm.zeros((1, n)))
    B -= nm.min(B)
    B /= nm.max(B)
    nm.fill_diagonal(B, nm.zeros((1, n)))

    for i in range(n-1):
        B[i, i+1:] = stats.bernoulli.rvs(B[i, i+1:],
                                         random_state=nm.random.RandomState())
        B[i+1:, i] = nm.transpose(B[i, i+1:])

    return nm.matrix(B)


def synthetic_modularity_matrix(A, eps):
    n = A.shape[1]
    K = nm.ones((1, n)) * A

    B = A - nm.transpose(K) * K / float(sum(nm.ravel(K)))
    x, l = leading_eigenvector(B)
    b = nm.transpose(K) * (K * x) / float(sum(nm.ravel(K))) + l * x

    return generate_matrix(x, nm.ravel(b), eps, nm.ravel(K))


def modularity_clone_graph(G, m=0.9):
    names = G.nodes()
    A = nx.to_numpy_matrix(G, nodelist=names)
    n = nm.shape(A)[0]
    B = modularity_clone_matrix(A, int(round(n*m)))
    H = gu.simm_matrix_2_graph(B, names)
    return H

if __name__ == "__main__":
    __eigen_info = True
    __eigen_debug = True
    info("Loading data..")
    G = nx.karate_club_graph()
    A = nx.to_numpy_matrix(G)
    B = synthetic_modularity_matrix(A, 0.0001)
    print A-B
