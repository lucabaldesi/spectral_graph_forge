#!/usr/bin/env python
import athina_net as an
import networkx as nx
import numpy as nm
import community
import graph_utils as gu
import eigen_graph as eg


def modgraph(G, level):
        A = nx.to_numpy_matrix(G)
        n = nm.shape(A)[0]
        B = eg.modularity_clone_matrix(A, level)
        H = gu.simm_matrix_2_graph(B)
#        while nx.is_isomorphic(G, H):
#            B = eg.modularity_clone_matrix(A, level)
#            H = gu.simm_matrix_2_graph(B)
        return H


def inc_alpha_test():
    n = 128
    inter_c_p = 0.001
    comm = 8
    outfile = "inc_alpha_" + str(comm) + "comm.data"
    part = { i: comm*i/128 for i in range(n)}
    with open(outfile, 'w') as f:
        f.write("Run,edge_prob,actual_mod,found_comm,louvain_mod,norm_mod,level,modgraph_mod,mod_ratio\n")
        for i in range(10):
            for level in range(1, n):
                for p in nm.arange(0.15, 0.95, 0.05):
                    print("{0:d} {1:d} {2:f}".format(i, level, p))
                    G = an.athina_net(inter_c_p, p, comm, n)
                    am = community.modularity(part, G)
                    m1 = community.modularity(community.best_partition(G), G)
                    nmod, ncomm = gu.norm_modularity(G)
                    H = modgraph(G, level)
                    m2 = community.modularity(community.best_partition(H), H)
                    f.write("{0:d},{1:f},{2:f},{3:d},{4:f},{5:f},{6:f},{7:f},{8:f}\n".format(i, p, am, ncomm, m1, nmod, level, m2, m2/m1))


if __name__ == "__main__":
    inc_alpha_test()
