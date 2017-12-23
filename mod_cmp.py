#!/usr/bin/env python
import os
import gc
import networkx as nx
import numpy as nm
import time

import eigen_graph as eg
import graph_utils as gu
import parallelism
import sah
import trajanovski as traj
import read_pajek as paj
import twok_helper as twok
import dc_sbm as sbm

from twok_vs_eigen import get_statistics1


def graph_worker_oneshot(inputlist, queue, print_queue):
    for duty in inputlist:
        name = duty[0]
        G = duty[1]
        algo = duty[2]
        param = duty[3]

        A = nx.to_numpy_matrix(G)
        x, l = eg.eigen_centrality(A)

        eg.info("Setup completed")
        start_time = time.time()

        if algo == "spectre":
            m = float(param)
            n = nm.shape(A)[0]
            B = eg.sample_simm_matrix2(A, int(round(n*m)))
            H = None
            algo += str(m)

        elif algo == "traj":
            H = traj.random_graph(G)

        elif algo == "sah":
            dists = param.split(",")
            H = sah.random_graph(G, dists[0], dists[1])

        elif algo == "modularity":
            m = float(param)
            n = nm.shape(A)[0]
            B = eg.modularity_clone_matrix(A, int(round(n*m)))
            H = gu.simm_matrix_2_graph(B)
            while nx.is_isomorphic(G, H):
                B = eg.modularity_clone_matrix(A, int(round(n*m)))
                H = gu.simm_matrix_2_graph(B)
            algo += str(m)

        elif algo == "25k":
            H = twok.gen_25k_graph(G)

        elif algo == "sbm":
            H = sbm.generate(G)

        eg.info("Graph Generated")

        gc.collect()
        stat = get_statistics1(G, H, time.time()-start_time, fraction=0.1)
        s = algo + "," + name + "," + str(len(G.nodes()))
        for el in stat:
            s += "," + str(el)
        print_queue.put(s)
        print_queue.put("\n")
        gc.collect()


def init_outfile(outfile):
    with open(outfile, 'w') as f:
        f.write("Strategy," + "Graph," + "Nodes," +
                str("EigErr") + "," + str("DegCorr") + "," +
                str("ClustRatio") + "," + str("EigVErr") + "," +
                str("NodeDistCorr") + "," + str("DegBetCorr") + "," +
                str("KCoreCorr") + "," + str("CommNeighDist") + "," +
                str("PartRatio") + "," + str("CommunityRatio") + "," +
                str("AvgNeighDegCorr") + "," + "CommSizeCorr" + "," +
                str("Connected") + "," + str("Duration") + "\n")


# def cmp_test():
#     eg.__eigen_info = True
#     outfile = "mod_cmp.data"
#     init_outfile(outfile)
#     shots = []
#
#     eg.info("Graph Generated")
#
#     gc.collect()
#     stat = get_statistics1(G, H, time.time()-start_time, fraction=0.1)
#     s = algo + "," + name + "," + str(len(G.nodes()))
#     for el in stat:
#         s += "," + str(el)
#     print_queue.put(s)
#     print_queue.put("\n")
#     gc.collect()


# def init_outfile(outfile):
#     with open(outfile, 'w') as f:
#         f.write("Strategy," + "Graph," + "Nodes," +
#                 str("EigErr") + "," + str("DegCorr") + "," +
#                 str("ClustRatio") + "," + str("EigVErr") + "," +
#                 str("NodeDistCorr") + "," + str("DegBetCorr") + "," +
#                 str("KCoreCorr") + "," + str("CommNeighDist") + "," +
#                 str("PartRatio") + "," + str("CommunityRatio") + "," +
#                 str("AvgNeighDegCorr") + "," + "CommSizeCorr" + "," +
#                 str("Connected") + "," + str("Duration") + "\n")


def cmp_test():
    eg.__eigen_info = True
    outfile = "mod_cmp2.data"
    init_outfile(outfile)
    shots = []

    graph_folder = "add_health/"
    for f in os.listdir(graph_folder):
        if f.endswith(".paj"):
            G = paj.read_pajek(graph_folder + f, edge_attribute=False)
            name = f.split(".")[0]
            shots += [(name, G, "modularity", 0.2)]
            shots += [(name, G, "modularity", 0.3)]
            shots += [(name, G, "modularity", 0.4)]
            shots += [(name, G, "modularity", 0.6)]
            shots += [(name, G, "modularity", 0.7)]
            shots += [(name, G, "modularity", 0.8)]
            # shots += [(name, G, "sbm", 0.9)]
            # shots += [(name, G, "25k", 0.9)]
            # shots += [(name, G, "traj", 0.9)]
            # shots += [(name, G, "modularity", 0.9)]
    graph_folder = "benchmark_2_1/newman_girvan/"
    for f in os.listdir(graph_folder):
        if f.endswith(".edges"):
            G = nx.read_edgelist(graph_folder + f)
            name = f.split(".")[0]
            shots += [(name, G, "modularity", 0.2)]
            shots += [(name, G, "modularity", 0.3)]
            shots += [(name, G, "modularity", 0.4)]
            shots += [(name, G, "modularity", 0.6)]
            shots += [(name, G, "modularity", 0.7)]
            shots += [(name, G, "modularity", 0.8)]
            # shots += [(name, G, "modularity", 0.9)]
            # shots += [(name, G, "sbm", 0.9)]
            # shots += [(name, G, "25k", 0.9)]
    graph_folder = "benchmark_2_1/fortunato/"
    for f in os.listdir(graph_folder):
        if f.endswith(".edges"):
            G = nx.read_edgelist(graph_folder + f)
            name = f.split(".")[0]
            shots += [(name, G, "modularity", 0.2)]
            shots += [(name, G, "modularity", 0.3)]
            shots += [(name, G, "modularity", 0.4)]
            shots += [(name, G, "modularity", 0.6)]
            shots += [(name, G, "modularity", 0.7)]
            shots += [(name, G, "modularity", 0.8)]
            # shots += [(name, G, "25k", 0.9)]
            # shots += [(name, G, "sbm", 0.9)]
    graph_folder = "facebook/"
    for f in os.listdir(graph_folder):
        if f.endswith(".edges"):
            G = nx.read_edgelist(graph_folder + f)
            name = "facebook"  # + f.split(".")[0]
            shots += [(name, G, "modularity", 0.2)]
            shots += [(name, G, "modularity", 0.3)]
            shots += [(name, G, "modularity", 0.4)]
            shots += [(name, G, "modularity", 0.6)]
            shots += [(name, G, "modularity", 0.7)]
            shots += [(name, G, "modularity", 0.8)]
            # shots += [(name, G, "25k", 0.9)]
            # shots += [(name, G, "sbm", 0.9)]
            # shots += [(name, G, "traj", 0.9)]
            # shots += [(name, G, "modularity", 0.9)]
    print("Shots loaded")
    parallelism.minions(shots*10, graph_worker_oneshot, parallelism=7,
                        outfile=outfile)


if __name__ == "__main__":
    cmp_test()
