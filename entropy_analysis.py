#!/usr/bin/env python

import networkx as nx
import numpy as nm
import scipy.stats as stats
import math
import os
# import time
import random
import subprocess as sproc

import eigen_graph as eg
import read_pajek as paj
import graph_utils as gu
import community


def truncated_mod_approx(A, level):
    n = A.shape[1]
    K = nm.ones((1, n)) * A
    B = A - nm.transpose(K) * K / float(sum(nm.ravel(K)))
    B = eg.mat_spect_approx(B, level, sorteigs=True, absolute=True)

    B = B + nm.transpose(K) * K / float(sum(nm.ravel(K)))

    B = nm.vectorize(gu.truncate, otypes=[nm.float])(B)
    nm.fill_diagonal(B, nm.zeros((1, n)))

    return nm.matrix(B)


def sample_mod_matrix(B):
    n = B.shape[1]
    B = nm.vectorize(eg.scaling_fun, otypes=[nm.float])(B)
    nm.fill_diagonal(B, nm.zeros((1, n)))

    for i in range(n-1):
        B[i, i+1:] = stats.bernoulli.rvs(B[i, i+1:])  # ,
        # random_state=nm.random.RandomState())
        B[i+1:, i] = nm.transpose(B[i, i+1:])
    return nm.matrix(B)


def mat_entropy(B):
    n = B.shape[1]
    h = 0
    d = 0  # expected density
    for i in range(n):
        for j in range(i, n):
            p = round(B[i, j], 14)
            d += p
            if p != 0 and p != 1:
                h -= p * math.log(p, 2)
                h -= (1 - p) * math.log(1 - p, 2)

    d /= float(n**2 - n)/2
    s = -float(n*(n-2))/(2*(math.log(d, 2) + math.log(1-d, 2)))
    # s = float(n**2 - n)/2
    return h/s


def deanon_ratio(G, H, seeds=0.05):
    nx.write_weighted_edgelist(G, "orig.edges")
    nx.write_weighted_edgelist(H, "new.edges")
    n = len(G.nodes())
    seed_nodes = random.sample(G.nodes(), int(round(seeds * n)))
    with open("seeds.txt", "w") as s:
        for sn in seed_nodes:
            s.write(str(sn) + " " + str(sn) + "\n")
    cmd = "java -jar secGraphCLI.jar -m d -a DV -gA orig.edges \
        -gB new.edges -seed seeds.txt -bSize " + str(n) + " -nKeep \
        " + str(n) + " -gO out.txt -am stack"
    res = sproc.check_output(cmd, shell=True)
    sval = res.split(" ")[2].split("/")
    return float(sval[0])/float(sval[1])


def entropy_analysis(G):
    with open("entropy.data", "w") as doc:
        doc.write("graph,alpha,modratio,entropy,denanonratio\n")
        for e in G:
            name = e[0]
            g = e[1]
            # gu.connect_components(g)  # WARNING
            for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
                names = g.nodes()
                A = nx.to_numpy_matrix(g, nodelist=names)
                n = min(A.shape)
                for i in range(10):
                    B = truncated_mod_approx(A, int(round(n*alpha)))
                    H = gu.simm_matrix_2_graph(sample_mod_matrix(B), names)
                    gu.connect_components(H)
                    modratio = (community.modularity(
                        community.best_partition(H), H) / community.modularity(
                            community.best_partition(g), g))
                    entropy = mat_entropy(B)
                    deanonratio = deanon_ratio(g, H)
                    doc.write(name + "," + str(alpha) + "," +
                              str(modratio) + "," +
                              str(entropy) + "," + str(deanonratio) + "\n")


if __name__ == "__main__":
    G = []
    G.append(("comm70", paj.read_pajek("add_health/comm70.paj", False)))
    G.append(("facebook107", nx.read_weighted_edgelist("facebook/348.edges")))
    # folder = "add_health/"
    # for f in os.listdir(folder):
    #     if f.endswith(".paj"):
    #         G.append((f.split(".")[0], paj.read_pajek(folder+f, False)))
    # folder = "facebook/"
    # for f in os.listdir(folder):
    #     if f.endswith(".edges"):
    #         G.append(("facebook"+f.split(".")[0], nx.read_weighted_edgelist(folder+f)))
    entropy_analysis(G)
