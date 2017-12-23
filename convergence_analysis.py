#!/usr/bin/env python

import networkx as nx
import numpy as nm
import scipy.stats as stats
import math

import eigen_graph as eg
import graph_utils as gu
import entropy_analysis as ea


def gen_networks(n, p, g=10):
    G = []
    m = int(round((n - math.sqrt(n**2 - p*2*(n**2 - n)))/2.))
    for i in range(g):
        G.append(("ER", nx.erdos_renyi_graph(n, p)))
        G.append(("BA", nx.barabasi_albert_graph(n, m)))
    with open("conv_eigv.data", "w") as doc:
        doc.write("type,id,eignum,eigvalue\n")
        for i in range(g):
            A = nx.to_numpy_matrix(G[i][1])
            d, V = nm.linalg.eigh(A)
            d = nm.ravel(d)
            k = nm.argsort(nm.abs(d))
            k = nm.flipud(k)
            for j in range(n):
                doc.write(G[i][0] + "," + str(i) + "," + str(j+1) + "," +
                          str(abs(d[k[j]])) + "\n")
    return G


def networkize(B, norm_func=None):
    n = min(B.shape)
    nm.fill_diagonal(B, nm.zeros((1, n)))
    coins = 0

    if norm_func:
        B = nm.vectorize(norm_func, otypes=[nm.float])(B)
        for i in range(n):
            for j in range(i-1):
                el = round(B[i, j], 10)
                if el < 1 and el > 0:
                    coins += 1
        coins /= float(n**2 - n)/2
    else:
        B -= nm.min(B)
        if nm.max(B) != 0:
            B /= nm.max(B)

    nm.fill_diagonal(B, nm.zeros((1, n)))

    for i in range(n-1):
        B[i, i+1:] = stats.bernoulli.rvs(B[i, i+1:])
        B[i+1:, i] = nm.transpose(B[i, i+1:])

    return nm.matrix(B), coins


def normalize(B, norm_func=None):
    n = min(B.shape)
    nm.fill_diagonal(B, nm.zeros((1, n)))
    coins = 0

    if norm_func:
        B = nm.vectorize(norm_func, otypes=[nm.float])(B)
        for i in range(n):
            for j in range(i-1):
                el = round(B[i, j], 10)
                if el < 1 and el > 0:
                    coins += 1
        coins /= float(n**2 - n)/2
    else:
        B -= nm.min(B)
        if nm.max(B) != 0:
            B /= nm.max(B)

    nm.fill_diagonal(B, nm.zeros((1, n)))
    return nm.matrix(B), coins


def sample(B):
    n = min(B.shape)
    nm.fill_diagonal(B, nm.zeros((1, n)))
    for i in range(n-1):
        B[i, i+1:] = stats.bernoulli.rvs(B[i, i+1:])
        B[i+1:, i] = nm.transpose(B[i, i+1:])

    return nm.matrix(B)


def synth_networks(G):
    with open("conv_err.data", "w") as doc:
        doc.write("type,id,eignum,appr_err,trunc_err,trunc_coins,trunc_entropy,logistic_err,logistic_entropy,eq_err,eq_entropy\n")
        j = 0
        for g in G:
            print "On graph: " + str(j) + " " + g[0]
            A = nx.to_numpy_matrix(g[1])
            n = min(A.shape)
            for i in range(n):
                B = nm.matrix(eg.mat_spect_approx(A, i+1, True, absolute=True))
                err1 = nm.linalg.norm(A-B, ord=2)

                C, trunc_coins = normalize(B, gu.truncate)
                trunc_entropy = ea.mat_entropy(C)
                C = sample(C)
                trunc_err = nm.linalg.norm(A-C, ord=2)

                C, coins = normalize(B, gu.logistic)
                logistic_entropy = ea.mat_entropy(C)
                C = sample(C)
                logistic_err = nm.linalg.norm(A-C, ord=2)

                C, coins = normalize(B)
                eq_entropy = ea.mat_entropy(C)
                C = sample(C)
                eq_err = nm.linalg.norm(A-C, ord=2)

                doc.write(g[0] + "," +
                          str(j) + "," + str(float(i+1)/n) + "," +
                          str(err1) + "," + str(trunc_err) + "," +
                          str(trunc_coins) + "," + str(trunc_entropy) + "," +
                          str(logistic_err) + "," + str(logistic_entropy) + "," +
                          str(eq_err) + "," + str(eq_entropy) + "\n")
            j += 1


if __name__ == "__main__":
    G = gen_networks(100, 0.1)
    synth_networks(G)
