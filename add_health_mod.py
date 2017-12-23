#!/usr/bin/env python


import os
import networkx as nx
import time
import gc
import numpy as nm

import graph_utils as gu
import read_pajek as paj
import eigen_graph as eg
import community
import parallelism
import trajanovski as traj

import twok_helper as twok
import dc_sbm as sbm


def add_health_partition(G, attr):
    part = {}
    for n in G.nodes():
        part[n] = G.node[n][attr]
    return part


def get_statistics(G, H, duration, fraction):
    mod2 = community.modularity(community.best_partition(H), H)
    mod1 = community.modularity(community.best_partition(G), G)
    wmod = mod2/mod1

    part = add_health_partition(G, 'sex')
    mod2 = community.modularity(part, H)
    mod1 = community.modularity(part, G)
    sexmod = mod2/mod1

    part = add_health_partition(G, 'race')
    mod2 = community.modularity(part, H)
    mod1 = community.modularity(part, G)
    racemod = mod2/mod1

    part = add_health_partition(G, 'grade')
    mod2 = community.modularity(part, H)
    mod1 = community.modularity(part, G)
    grademod = mod2/mod1

    return (wmod, sexmod, racemod, grademod)


def graph_worker_oneshot(inputlist, queue, print_queue):
    for duty in inputlist:
        name = duty[0]
        G = duty[1]
        algo = duty[2]
        param = duty[3]

        names = G.nodes()
        names.sort(key=int)
        A = nx.to_numpy_matrix(G, nodelist=names)

        eg.info("Setup completed")
        start_time = time.time()

        if algo == "modularity":
            m = float(param)
            n = nm.shape(A)[0]
            B = eg.modularity_clone_matrix(A, int(round(n*m)))
            H = gu.simm_matrix_2_graph(B, names)
            while nx.is_isomorphic(G, H):
                B = eg.modularity_clone_matrix(A, int(round(n*m)))
                H = gu.simm_matrix_2_graph(B, names)
            algo += str(m)

        elif algo == "traj":
            H = traj.random_graph(G)
            trnames = H.nodes()
            mapping = {trnames[i]: names[i] for i in range(len(names))}
            H = nx.relabel_nodes(H, mapping)

        elif algo == "25k":
            H = twok.gen_25k_graph(G, names)

        elif algo == "sbm":
            H = sbm.generate(G, names)

        eg.info("Graph Generated " + name)

        gc.collect()
        stat = get_statistics(G, H, time.time()-start_time, fraction=0.1)
        s = name + "," + str(len(G.nodes())) + "," + algo
        for el in stat:
            s += "," + str(el)
        print_queue.put(s)
        print_queue.put("\n")
        gc.collect()


def init_outfile(out):
    with open(out, 'w') as f:
        f.write("Graph,Nodes,Algo,\
WModRatio,SexModRatio,RaceModRatio,GradeModRatio\n")


def add_health_mod():
    eg.__eigen_info = True
    outfile = "add_health_mod2.data"
    init_outfile(outfile)
    shots = []

    graph_folder = "add_health/"
    for f in os.listdir(graph_folder):
        if f.endswith(".paj"):
            G = paj.read_pajek(graph_folder + f, edge_attribute=False)
            name = f.split(".")[0]
            shots += [(name, G, "modularity", 0.1)]
            shots += [(name, G, "modularity", 0.5)]
            shots += [(name, G, "modularity", 0.75)]
            # shots += [(name, G, "modularity", 0.9)]
            # shots += [(name, G, "traj", 0.9)]
            # shots += [(name, G, "25k", 0.9)]
            # shots += [(name, G, "sbm", 0.9)]
    print("Shots loaded")
    parallelism.minions(shots*10, graph_worker_oneshot, parallelism=7,
                        outfile=outfile)


if __name__ == "__main__":
    add_health_mod()
