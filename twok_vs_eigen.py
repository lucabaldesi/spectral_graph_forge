#!/usr/bin/env python

from __future__ import print_function
import sys
import os
import random
import networkx as nx
import numpy as nm
import community
import time
import gc

import eigen_graph as eg
import graph_utils as gu
sys.path.append("twoK")
from twok_simple import joint_degree_graph
sys.path.append("twofiveK")
from Estimation import Estimation
from Generation import Generation

from net_formats import read_graphml, read_mat_file
import parallelism


def write_statistics(A, B, label, net, x, l, output=True):
    eg.info("Computing new centrality..")
    G = nx.from_numpy_matrix(A)
    H = nx.from_numpy_matrix(B)
    nx.write_weighted_edgelist(H, net + "_" + label + "_" +
                               str(random.randint(0, 99999999)) + ".edges")
    y, m = eg.eigen_centrality(B, maxiter=100000)

    eg.info("Printing out results..")
    eigen_err = eg.vec_dist(x, y)
    clust_err = nm.average(nx.clustering(G).values()) /\
        nm.average(nx.clustering(H).values())
    lambda_err = abs(l / m)

    degree_corr = gu.correlation(sorted(nx.degree(G)), sorted(nx.degree(H)))
    if nx.is_connected(H):
        conn = 1
    else:
        conn = 0
    out = (str(label) + "," + str(net.split(".")[0].split("/")[-1]) + "," +
           str(eigen_err) + "," + str(degree_corr) + "," +
           str(clust_err) + "," + str(lambda_err) + "," +
           str(-1) + "," + str(-1) + "," +
           #  str(gu.correlate_dist_dict(gu.node_distance_dist(A),
           #                             gu.node_distance_dist(B))) + "," +
           #  str(gu.correlate_dist_dict(gu.get_degree_betweeness(A),
           #                             gu.get_degree_betweeness(B))) + "," +
           str(gu.correlate_dist_dict(gu.get_kcoreness(A),
                                      gu.get_kcoreness(B))) + "," +
           str(gu.correlate_dist_dict(gu.get_common_neigh_dist(A),
                                      gu.get_common_neigh_dist(B))) + "," +
           str(-1) + "," +
           #  str(community.modularity(
           #      community.best_partition(G), G) /
           #      community.modularity(
           #      community.best_partition(H), H)) + "," +
           str(gu.correlate_dist_dict(gu.get_avg_neighbour_degree(A),
                                      gu.get_avg_neighbour_degree(B))) + "," +
           str(conn))
    if output:
        print(out, file=sys.stderr)
    return out


def csv_test_unparallel():
    eg.__eigen_info = True
    print("Strategy," + "Graph," +
          str("EigErr") + "," + str("DegCorr") + "," +
          str("ClustRatio") + "," + str("EigVErr") + "," +
          str("NodeDistCorr") + "," + str("DegBetCorr") + "," +
          str("KCoreCorr") + "," + str("CommNeighDist") + "," +
          str("PartRatio") + "," + str("AvgNeighDegCorr") + "," +
          str("Connected"), file=sys.stderr)
    for filo in os.listdir("/home/baldo/tmp/graph_generator/PL200/"):
        with open("/home/baldo/tmp/graph_generator/PL200/" + filo, "r") as net:
            eg.info(filo)
            eg.info("Loading graph..")
            G = nx.read_weighted_edgelist(net)  # , delimiter=":")
            # G = read_graphml(net)
            A = nx.to_numpy_matrix(G)
            n = nm.shape(A)[0]
            joint_degrees = nx.algorithms.mixing.degree_mixing_dict(G)
            eg.info("Computing centrality..")
            x, l = eg.eigen_centrality(A)

            for i in range(10):
                eg.info("Run: " + str(i))

                eg.info("Building JDM graph..")
                H = joint_degree_graph(joint_degrees)
                B = nx.to_numpy_matrix(H)
                write_statistics(A, B, "2k", filo,
                                 x, l)

                eg.info("Building degree sequence graph..")
                H = nx.random_degree_sequence_graph((nx.degree(G).values()))
                B = nx.to_numpy_matrix(H)
                write_statistics(A, B, "1k", filo,
                                 x, l)

                precision = 0.01
                eg.info("Building eigen " + str(precision) + " graph..")
                B = eg.build_matrix(x, l, precision)
                write_statistics(A, B, "eig" + str(precision),
                                 filo, x, l)

                precision = 0.001
                eg.info("Building eigen " + str(precision) + " graph..")
                B = eg.build_matrix(x, l, precision)
                write_statistics(A, B, "eig" + str(precision),
                                 filo, x, l)

                precision = 0.0001
                eg.info("Building eigen " + str(precision) + " graph..")
                B = eg.build_matrix(x, l, precision)
                write_statistics(A, B, "eig" + str(precision),
                                 filo, x, l)

                m = 0.25
                eg.info("Building spectral " + str(m) + " graph..")
                B = eg.sample_simm_matrix(A, int(round(n*m)))
                write_statistics(A, B, "spectre" + str(m),
                                 filo, x, l)

                m = 0.5
                eg.info("Building spectral " + str(m) + " graph..")
                B = eg.sample_simm_matrix(A, int(round(n*m)))
                write_statistics(A, B, "spectre" + str(m),
                                 filo, x, l)

                m = 0.75
                eg.info("Building spectral " + str(m) + " graph..")
                B = eg.sample_simm_matrix(A, int(round(n*m)))
                write_statistics(A, B, "spectre" + str(m),
                                 filo, x, l)

                m = 0.9
                eg.info("Building spectral " + str(m) + " graph..")
                B = eg.sample_simm_matrix(A, int(round(n*m)))
                write_statistics(A, B, "spectre" + str(m),
                                 filo, x, l)

                m = 0.95
                eg.info("Building spectral " + str(m) + " graph..")
                B = eg.sample_simm_matrix(A, int(round(n*m)))
                write_statistics(A, B, "spectre" + str(m),
                                 filo, x, l)

                eg.info("Building D2.5 graph..")
                test25 = Estimation()
                gen25 = Generation()
                test25.load_graph("", graph=G)
                test25.calcfull_CCK()
                test25.calcfull_JDD()
                gen25.set_JDD(test25.get_JDD('full'))
                gen25.set_KTRI(test25.get_KTRI('full'))
                gen25.construct_triangles_2K()
                gen25.mcmc_improved_2_5_K(error_threshold=0.05)
                H = gen25.G
                B = nx.to_numpy_matrix(H)
                write_statistics(A, B, "25k", filo, x, l)


def graph_worker(inputlist, queue, print_queue):
    for filo in inputlist:
        if filo.split(".")[-1] == "graphml":
            G = read_graphml(filo)
        else:
            G = nx.read_weighted_edgelist(filo)

        A = nx.to_numpy_matrix(G)
        n = nm.shape(A)[0]
        joint_degrees = nx.algorithms.mixing.degree_mixing_dict(G)
        x, l = eg.eigen_centrality(A)

        H = nx.random_degree_sequence_graph((nx.degree(G).values()))
        B = nx.to_numpy_matrix(H)
        print_queue.put(write_statistics(A, B, "1k", filo,
                        x, l, output=False))
        print_queue.put("\n")

        H = joint_degree_graph(joint_degrees)
        B = nx.to_numpy_matrix(H)
        print_queue.put(write_statistics(A, B, "2k", filo,
                        x, l, output=False))
        print_queue.put("\n")

        precision = 0.01
        B = eg.build_matrix(x, l, precision)
        print_queue.put(write_statistics(A, B, "eig" + str(precision),
                        filo, x, l, output=False))
        print_queue.put("\n")

        precision = 0.001
        B = eg.build_matrix(x, l, precision)
        print_queue.put(write_statistics(A, B, "eig" + str(precision),
                        filo, x, l, output=False))
        print_queue.put("\n")

        precision = 0.0001
        B = eg.build_matrix(x, l, precision)
        print_queue.put(write_statistics(A, B, "eig" + str(precision),
                        filo, x, l, output=False))
        print_queue.put("\n")

        m = 0.25
        B = eg.sample_simm_matrix(A, int(round(n*m)))
        print_queue.put(write_statistics(A, B, "spectre" + str(m),
                        filo, x, l, output=False))
        print_queue.put("\n")

        m = 0.5
        B = eg.sample_simm_matrix(A, int(round(n*m)))
        print_queue.put(write_statistics(A, B, "spectre" + str(m),
                        filo, x, l, output=False))
        print_queue.put("\n")

        m = 0.75
        B = eg.sample_simm_matrix(A, int(round(n*m)))
        print_queue.put(write_statistics(A, B, "spectre" + str(m),
                        filo, x, l, output=False))
        print_queue.put("\n")

        m = 0.9
        B = eg.sample_simm_matrix(A, int(round(n*m)))
        print_queue.put(write_statistics(A, B, "spectre" + str(m),
                        filo, x, l, output=False))
        print_queue.put("\n")

        m = 0.95
        B = eg.sample_simm_matrix(A, int(round(n*m)))
        print_queue.put(write_statistics(A, B, "spectre" + str(m),
                        filo, x, l, output=False))
        print_queue.put("\n")

        test25 = Estimation()
        gen25 = Generation()
        test25.load_graph("", graph=G)
        test25.calcfull_CCK()
        test25.calcfull_JDD()
        gen25.set_JDD(test25.get_JDD('full'))
        gen25.set_KTRI(test25.get_KTRI('full'))
        gen25.construct_triangles_2K()
        gen25.mcmc_improved_2_5_K(error_threshold=0.05)
        H = gen25.G
        B = nx.to_numpy_matrix(H)
        print_queue.put(write_statistics(A, B, "25k", filo, x, l, output=False))
        print_queue.put("\n")


def stat_worker(inputlist, outqueue, print_queue):
    for el in inputlist:
        alg = el[0]
        A = el[1]
        B = el[2]
        if alg == "clust_ratio":
            a = nm.average(nx.clustering(A).values())
            b = nm.average(nx.clustering(B).values())
            v = a/b
        if alg == "degree_corr":
            v = gu.correlation(sorted(nx.degree(A)),
                               sorted(nx.degree(B)))
        if alg == "dist_dist":
            v = gu.correlate_dist_dict(gu.node_distance_dist(A),
                                       gu.node_distance_dist(B))
        if alg == "deg_bet_corr":
            v = gu.correlate_dist_dict(gu.get_degree_betweeness(A),
                                       gu.get_degree_betweeness(B))
        if alg == "kcore_corr":
            v = gu.correlate_dist_dict(gu.get_kcoreness(A),
                                       gu.get_kcoreness(B))
        if alg == "comm_neigh_corr":
            v = gu.correlate_dist_dict(gu.get_common_neigh_dist(A),
                                       gu.get_common_neigh_dist(B))
        if alg == "mod_ratio":
            v = community.modularity(community.best_partition(A), A) /\
                community.modularity(community.best_partition(B), B)
        if alg == "avg_neigh_deg_corr":
            v = gu.correlate_dist_dict(gu.get_avg_neighbour_degree(A),
                                       gu.get_avg_neighbour_degree(B))
        eg.info("Computed " + alg)
        outqueue.put({alg: v})


def get_statistics2(G, H, A, B, x, l):
    eg.info("Computing statistics...")
    if not H:
        eg.info("Building networkx graph...")
        H = gu.simm_matrix_2_graph(B)
        gu.connect_components(H)

    eg.info("Computing centrality...")
    y, m = eg.eigen_centrality(B, maxiter=100000)
    eg.info("Computing lambda ratio...")
    lambda_err = abs(l / m)
    eg.info("Computing centrality distance...")
    eigen_err = eg.vec_dist(x, y)

    inputs = [("avg_neigh_deg_corr", A, B), ("mod_ratio", G, H),
              ("comm_neigh_corr", A, B), ("kcore_corr", A, B),
              ("deg_bet_corr", A, B), ("dist_dist", A, B),
              ("degree_corr", G, H), ("clust_ratio", G, H)]
    mets = parallelism.launch_workers(inputs, stat_worker,
                                      inputs_per_worker=1, parallelism=4)
    res = {}
    for el in mets:
        res.update(el)

    eg.info("Check connectivity...")
    if nx.is_connected(H):
        conn = 1
    else:
        conn = 0

    eg.info("Done with stats")
    return (eigen_err, res['degree_corr'], res['clust_ratio'], lambda_err,
            res['dist_dist'], res['deg_bet_corr'],
            res['kcore_corr'], res['comm_neigh_corr'],
            res['mod_ratio'], res['avg_neigh_deg_corr'],
            conn)


def get_statistics1(G, H, duration, fraction=0.5):
    sample_size = int(round(fraction*len(G.nodes())))
    gsample = random.sample(G.nodes(), sample_size)
    hsample = random.sample(H.nodes(), sample_size)

    eg.info("Computing statistics...")
    eg.info("Computing centrality...")
    x = nx.eigenvector_centrality_numpy(G)
    y = nx.eigenvector_centrality_numpy(H)

    eg.info("Computing centrality distance...")
    eigen_err = gu.sorted_correlation(x.values(), y.values())

    eg.info("Computing clustering ratio...")
    clust1 = nx.clustering(G, gsample)
    clust2 = nx.clustering(H, hsample)
    clust_err = sum(clust2.values())/sum(clust1.values())

    lambda_err = -1

    eg.info("Computing degree correlation...")
    degdist1 = {}
    degdist2 = {}
    degree1 = nx.degree(G, nbunch=gsample)
    degree2 = nx.degree(H, nbunch=hsample)
    for d in degree1.values():
        degdist1[d] = degdist1.get(d, 0) + 1
    for d in degree2.values():
        degdist2[d] = degdist2.get(d, 0) + 1
    print(degdist1)
    print(degdist2)
    degree_corr = gu.correlate_dist_dict(degdist1, degdist2)
    print(degree_corr)

    eg.info("Check connectivity...")
    if nx.is_connected(H):
        conn = 1
    else:
        conn = 0

    eg.info("Distance distribution correlation...")
    dist1 = {}
    dist2 = {}
    for i in range(sample_size):
        for j in range(i+1, sample_size):
            try:
                d = nx.algorithms.bidirectional_dijkstra(G, gsample[i],
                                                         gsample[j])[0]
                dist1[d] = dist1.get(d, 0) + 1
            except:
                pass
            try:
                d = nx.algorithms.bidirectional_dijkstra(H, hsample[i],
                                                         hsample[j])[0]
                dist2[d] = dist2.get(d, 0) + 1
            except:
                pass
    distance_dist_corr = gu.correlate_dist_dict(dist1, dist2)

    eg.info("Betweenness correlation...")
    b1 = nx.betweenness_centrality(G, sample_size).values()
    b2 = nx.betweenness_centrality(H, sample_size).values()
    bet_corr = gu.sorted_correlation(b1, b2)

    eg.info("K-coreness correlation...")
    kcore_corr = -1  # gu.correlate_dist_dict(gu.get_kcoreness(A),
    # gu.get_kcoreness(B))
    eg.info("Common neighbourhood correlation...")
    # comm1 = {}
    # comm2 = {}
    # for i in range(sample_size):
    #     for j in range(i+1, sample_size):
    #         d = len(list(nx.common_neighbors(G, gsample[i], gsample[j])))
    #         comm1[d] = comm1.get(d, 0) + 1
    #         d = len(list(nx.common_neighbors(H, hsample[i], hsample[j])))
    #         comm2[d] = comm2.get(d, 0) + 1
    common_neigh_corr = -1  # gu.correlate_dist_dict(comm1, comm2)

    eg.info("Modularity ratio...")
    Gm = gu.norm_modularity(G)
    Hm = gu.norm_modularity(H)
    modularity_ratio = Hm[0]/Gm[0]
    partition_ratio = Hm[1]/float(Gm[1])

    eg.info("Community size correlation...")
    gsize = gu.community_size(G)
    hsize = gu.community_size(H)
    comm_size_corr = gu.sorted_correlation(gsize, hsize)

    eg.info("Avg neighbourhood degree correlation...")
    avg1 = nx.average_neighbor_degree(G, nodes=gsample)
    avg2 = nx.average_neighbor_degree(H, nodes=hsample)
    avg_neigh_deg_corr = gu.sorted_correlation(avg1.values(), avg2.values())

    eg.info("Done with stats")
    return (eigen_err, degree_corr, clust_err, lambda_err,
            distance_dist_corr, bet_corr, kcore_corr, common_neigh_corr,
            modularity_ratio, partition_ratio, avg_neigh_deg_corr,
            comm_size_corr, conn, duration)


def get_statistics(G, H, A, B, x, l, duration):
    eg.info("Computing statistics...")
    if not H:
        eg.info("Building networkx graph...")
        H = gu.simm_matrix_2_graph(B)
    eg.info("Computing centrality...")
    y, m = (-1, -1)  # eg.eigen_centrality(B, maxiter=100000)
    #  nx.write_weighted_edgelist(H,  "pgp_spectre0.9_" +
    #                            str(random.randint(0, 99999999)) + ".edges")

    eg.info("Computing centrality distance...")
    eigen_err = eg.vec_dist(x, y)
    eg.info("Computing clustering ratio...")
    clust_err = gu.average_clustering(A) / gu.average_clustering(B)
    # clust_err = nm.average(nx.clustering(G).values()) /\
    #     nm.average(nx.clustering(H).values())
    eg.info("Computing lambda ratio...")
    lambda_err = abs(l / m)
    eg.info("Computing degree correlation...")
    degree_corr = gu.correlation(gu.get_degrees(A), gu.get_degrees(B))
    eg.info("Check connectivity...")
    if nx.is_connected(H):
        conn = 1
    else:
        conn = 0

    eg.info("Distance distribution correlation...")
    distance_dist_corr = -1  # gu.correlate_dist_dict(gu.node_distance_dist(A),
    #                    gu.node_distance_dist(B))
    eg.info("Degree betweenness correlation...")
    degree_bet_corr = -1  # gu.correlate_dist_dict(gu.get_degree_betweeness(A),
    #                 gu.get_degree_betweeness(B))
    eg.info("K-coreness correlation...")
    kcore_corr = -1  # gu.correlate_dist_dict(gu.get_kcoreness(A),
    #                    gu.get_kcoreness(B))
    eg.info("Common neighbourhood correlation...")
    common_neigh_corr = -1  # gu.correlate_dist_dict(gu.get_common_neigh_dist(A)
    #                    ,gu.get_common_neigh_dist(B))
    eg.info("Modularity ratio...")
    Gm = gu.norm_modularity(G)
    Hm = gu.norm_modularity(H)
    modularity_ratio = Gm[0]/Hm[0]
    partition_ratio = Gm[1]/float(Hm[1])
    eg.info("Avg neighbourhood degree correlation...")
    avg_neigh_deg_corr = -1  # gu.correlate_dist_dict(
    # gu.get_avg_neighbour_degree(A),  gu.get_avg_neighbour_degree(B))
    eg.info("Done with stats")
    return (eigen_err, degree_corr, clust_err, lambda_err,
            distance_dist_corr, degree_bet_corr, kcore_corr, common_neigh_corr,
            modularity_ratio, partition_ratio, avg_neigh_deg_corr,
            conn, duration)


def graph_worker_oneshot(inputlist, queue, print_queue):
    for duty in inputlist:
        name = duty[0]
        G = duty[1]
        algo = duty[2]
        param = duty[3]

        A = nx.to_numpy_matrix(G)

        eg.info("Setup completed")
        start_time = time.time()

        if algo == "1k":
            H = nx.random_degree_sequence_graph((nx.degree(G).values()))
            # B = nx.to_numpy_matrix(H)

        elif algo == "2k":
            joint_degrees = nx.algorithms.mixing.degree_mixing_dict(G)
            H = joint_degree_graph(joint_degrees)
            # B = nx.to_numpy_matrix(H)

        elif algo == "eig":
            precision = float(param)
            # B = eg.build_matrix(x, l, precision)
            x, l = eg.eigen_centrality(A)
            B = eg.generate_matrix(x, l*x, precision, gu.get_degrees(A))
            H = None
            algo += str(precision)

        elif algo == "modeig":
            precision = float(param)
            B = eg.synthetic_modularity_matrix(A, precision)
            H = None
            algo += str(precision)

        elif algo == "spectre":
            m = float(param)
            n = nm.shape(A)[0]
            B = eg.sample_simm_matrix2(A, int(round(n*m)))
            H = gu.simm_matrix_2_graph(B)
            while nx.is_isomorphic(G, H):
                B = eg.sample_simm_matrix2(A, int(round(n*m)))
                H = gu.simm_matrix_2_graph(B)
            algo += str(m)

        elif algo == "laplacian":
            m = float(param)
            n = nm.shape(A)[0]
            B = eg.laplacian_clone_matrix(A, int(round(n*m)))
            H = gu.simm_matrix_2_graph(B)
            while nx.is_isomorphic(G, H):
                B = eg.sample_simm_matrix2(A, int(round(n*m)))
                H = gu.simm_matrix_2_graph(B)
            algo += str(m)

        elif algo == "modspec":
            m = float(param)
            n = nm.shape(A)[0]
            B = eg.modspec_clone_matrix(A, int(round(n*m)))
            H = None
            algo += str(m)

        elif algo == "franky":
            m = float(param)
            n = nm.shape(A)[0]
            B = eg.franky_clone_matrix(A, int(round(n*m)))
            H = None
            algo += str(m)

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
            test25 = Estimation()
            gen25 = Generation()
            test25.load_graph("", graph=G)
            test25.calcfull_CCK()
            test25.calcfull_JDD()
            gen25.set_JDD(test25.get_JDD('full'))
            gen25.set_KTRI(test25.get_KTRI('full'))
            gen25.construct_triangles_2K()
            gen25.mcmc_improved_2_5_K(error_threshold=0.05)
            H = gen25.G
            # B = nx.to_numpy_matrix(H)

        eg.info("Graph Generated")

        stat = get_statistics1(G, H, time.time()-start_time)
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


def csv_test():
    eg.__eigen_info = False
    #  folder = "/home/baldo/Lavoro/UCI/brains/"
    #  files = os.listdir(folder)
    #  files = [folder + f for f in files]
    #  files *= 10

    #  files = ["pgp.edges"]

    outfile = "fb_net.data"
    init_outfile(outfile)

    nets = ["fb_net/Michigan23.mat", "fb_net/MSU24.mat",
            "fb_net/UIllinios20.mat"]  # ["fb_net/Reed98.mat",
    # "fb_net/Caltech36.mat", "fb_net/Simmons81.mat"]
    # , "fb_net/Oberlin44.mat", "fb_net/Howard90.mat",
    # "fb_net/Rice31.mat", "fb_net/Wake73.mat", "fb_net/UC64.mat",
    # "fb_net/UCSC68.mat", "fb_net/Duke14.mat", "fb_net/UCF52.mat",
    # "fb_net/Harvard1.mat" ]

    shots = []
    for net in nets:
        G = read_mat_file(net)
        name = net.split("/")[1]
        name = name.split(".")[0]
        shots += [(name, G, "modularity", 0.1)]
        shots += [(name, G, "spectre", 0.1)]
        shots += [(name, G, "modularity", 0.5)]
        shots += [(name, G, "spectre", 0.5)]
        shots += [(name, G, "modularity", 0.9)]
        shots += [(name, G, "spectre", 0.9)]
        # shots += [(name, G, "25k", 0.9)]
    print("Shots loaded")
    parallelism.minions(shots*10, graph_worker_oneshot, parallelism=4,
                        outfile=outfile)

if __name__ == "__main__":
    csv_test()
