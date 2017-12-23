import math
import random
import networkx as nx

import community


def scale_int_dict(x, n):
    el = x.copy()
    s = sum(el.values())
    for k in el:
        el[k] = int(round(n*el[k]/float(s)))
        if el[k] < 1:
            el[k] = 1
    s = sum(el.values())
    i = iter(el)
    while s != n:
        try:
            k = i.next()
        except StopIteration:
            i = iter(el)
            k = i.next()
        if s > n and el[k] > 1:
            el[k] -= 1
            s -= 1
        else:
            el[k] += 1
            s += 1
    return el


class CommGraph(object):

    def to_networkx(self, n):
        G = nx.Graph()
        E = []
        tot_nodes = 0
        comm_nodes = {}
        nodes = scale_int_dict(self.Lin, n)
        for i in range(self.c):
            e = nx.gnm_random_graph(nodes[i], self.Lin[i]).edges()
            E += [(ii + tot_nodes, jj + tot_nodes) for ii, jj in e]
            comm_nodes[i] = (tot_nodes, tot_nodes+nodes[i]-1)
            tot_nodes += nodes[i]

        for ci, cj in self.comm_links:
            i = random.randint(comm_nodes[ci][0], comm_nodes[ci][1])
            j = random.randint(comm_nodes[cj][0], comm_nodes[cj][1])
            E.append((i, j))

        G.add_edges_from(E)
        return G

    def num_links(self):
        return sum(self.Lin.values()) + sum(self.Lout.values())/2

    def community_degree(self):
        D = {}
        for c in self.Lin:
            D[c] = 2*self.Lin[c] + self.Lout[c]
        return D

    def min_tranf1(self):
        D = self.community_degree()
        L = self.num_links()
        i = max(D, key=D.get)
        j = min(D, key=D.get)
        value = (2*L + D[j] - D[i] - 1)/(2. * L**2)
        return (i, j, value)

    def min_inside_out(self):
        D = self.community_degree()
        L = self.num_links()
        i = max(D, key=D.get)
        F = {}
        for k,v in D.items():
            if k != i and self.Lin[k] > 1:
                F[k] = v
        j = min(F, key=F.get)
        value = float(2*L + D[j] - D[i] - 1)/(2. * (L**2))
        return (i, j, value)

    def min_outside_in(self):
        D = self.community_degree()
        L = self.num_links()
        i = max(D, key=D.get)
        F = {}
        for k,v in D.items():
            if k != i and self.num_inter_edges(i, k) > 0:
                F[k] = v
        j = min(F, key=F.get)
        value = (2*L + D[j] - D[i] - 1)/(2. * L**2)
        return (i, j, value)

    def min_tranf2(self):
        D = self.community_degree()
        L = self.num_links()
        i = max(D, key=D.get)
        j = min(D, key=D.get)
        if D[i] - D[j] > 2:
            value = (D[i] - D[j] - 2)/float(L**2)
        else:
            value = 0
        return (i, j, value)

    def comm_swap_edge_equalize(self):
        D = self.community_degree()
        L = self.num_links()
        i = max(D, key=D.get)
        F = {}
        for k,v in D.items():
            if k != i and self.Lin[k] > 1:
                F[k] = v
        j = min(F, key=F.get)
        if D[i] - D[j] > 2:
            value = (D[i] - D[j] - 2)/float(L**2)
        else:
            value = 0
        return (i, j, value)

    def comm_swap_edge_disequalize(self):
        D = self.community_degree()
        L = self.num_links()
        j = min(D, key=D.get)
        F = {}
        for k,v in D.items():
            if k != j and self.Lin[k] > 1:
                F[k] = v
        i = max(F, key=F.get)
        if D[i] - D[j] > 2:
            value = (D[i] - D[j] - 2)/float(L**2)
        else:
            value = 0
        return (i, j, value)

    def num_inter_edges(self, i, j):
        m = min(i, j)
        M = max(i, j)
        return self.comm_links.count((m, M))

    def remove_inter_edge(self, i, j):
        m = min(i, j)
        M = max(i, j)
        self.comm_links.remove((m, M))

    def add_inter_edge(self, i, j):
        if i <= self.c and j <= self.c and i != j:
            m = min(i, j)
            M = max(i, j)
            self.comm_links.append((m, M))
            self.Lout[i] = self.Lout.get(i, 0) + 1
            self.Lout[j] = self.Lout.get(j, 0) + 1

    def set_intra_edges(self, community, k):
        self.Lin[community] = k

    def dec_intra_edges(self, community):
        if self.Lin[community] > 0:
            self.Lin[community] -= 1

    def inc_intra_edges(self, community):
        self.Lin[community] += 1

    def modularity(self):
        D = self.community_degree()
        L = self.num_links()
        Lint = sum(self.Lout.values())/2
        s = 0.
        for j in range(self.c):
            for k in range(self.c):
                s += ((D[j] - D[k])/(2.*L))**2
        return 1 - 1./self.c - float(Lint)/L - s/(2.*self.c)

    def __init__(self, L, c):
        self.comm_links = []
        self.Lin = {}
        self.Lout = {}
        self.c = 0
        self.m = None
        self.m_max = 0
        self.c = c
        k = int(math.floor(L/float(self.c)))

        for i in range(self.c-1):
            self.Lin[i] = k-1
            self.add_inter_edge(i, i+1)
        self.Lin[i+1] = k-1

        i = 0
        while self.num_links() < L:
            self.inc_intra_edges(i)
            i += 1
            i %= c
        self.m = self.modularity()
        self.m_max = self.m
#         r = L % self.c
#         self.m_max = 1 - 1/float(self.c) - (self.c-1)/float(L)
#         self.Lin[1] = k
#         i = 2
#
#         if r == 0:
#             while i <= self.c:
#                 self.add_inter_edge(i-1, i)
#                 self.set_intra_edges(i, k-1)
#                 i = i + 1
#             self.m_max = self.m_max - 1/float(2 * (L**2))
#         elif r <= math.floor(self.c/2.):
#             while i <= self.c-r:
#                 self.add_inter_edge(i-1, i)
#                 self.set_intra_edges(i, k-1)
#                 if i <= r:
#                     self.add_inter_edge(self.c-i+1, i)
#                     self.set_intra_edges(self.c-i+1, k)
#                 i += 1
#             self.set_intra_edges(i, k)
#             self.m_max -= r*(self.c-2*r)/(2.*self.c*(L**2))
#         else:
#             while i <= r:
#                 self.add_inter_edge(i-1, i)
#                 self.set_intra_edges(i, k)
#                 if i <= self.c-r:
#                     self.add_inter_edge(self.c-i+1, i)
#                     self.dec_inter_edges(i)
#                     self.set_intra_edges(self.c-i+1, k)
#                 i += 1
#             self.m_max -= (self.c-r)*(2*r - self.c)/(2.*self.c*(L**2))
#
#         self.m = self.m_max
#         print self.num_links()
        assert(self.num_links() == L)


class TMGG(object):
    def __init__(self, L, c, m, eps=0.001):
        self.target_m = 0
        self.state = 0
        self.delta_m_cur = float("inf")
        self.eps = 0
        self.com_graph = CommGraph(L, c)
        self.target_m = m
        self.eps = eps

    def get_random_graph(self, n):
        self.optimize_modularity()
        self.G = self.com_graph.to_networkx(n)
        return self.G

    def increase_modularity1(self):
        moving = False
        if self.com_graph.m > self.target_m:  # we must reduce modularity
            i, j, value = self.com_graph.min_inside_out()
            assert(value > 0)
            self.state = 1
            if self.com_graph.Lin[j] > 1:
                self.com_graph.dec_intra_edges(j)
                self.com_graph.add_inter_edge(i, j)
                # if abs(self.com_graph.modularity() - self.com_graph.m + value) > self.eps:
                #     print abs(self.com_graph.modularity() - self.com_graph.m + value)
                # assert(abs(self.com_graph.modularity() - self.com_graph.m + value) < self.eps)
                self.com_graph.m -= value
                self.com_graph.m = self.com_graph.modularity()  # NEEDED as formula doesn't work
                moving = True
        else: # we must increase modularity
            self.state = 2
            i, j, value = self.com_graph.min_outside_in()
            assert(value > 0)
            if self.com_graph.num_inter_edges(i, j) > 0:
                self.com_graph.remove_inter_edge(i, j)
                self.com_graph.inc_intra_edges(j)
                self.com_graph.m += value
                self.com_graph.m = self.com_graph.modularity()  # NEEDED as formula doesn't work
                moving = True
        return moving

    def increase_modularity2(self):
        moving = False
        if self.com_graph.m > self.target_m:  # we must reduce modularity
            self.state = 1
            i, j, value = self.com_graph.comm_swap_edge_equalize()
            assert(value >= 0)
            if value > 0 and self.com_graph.Lin[j] > 1:
                self.com_graph.dec_intra_edges(j)
                self.com_graph.m -= value
                self.com_graph.m = self.com_graph.modularity()  # NEEDED as formula doesn't work
                self.com_graph.inc_intra_edges(i)
                moving = True
        else:  # we must increase modularity
            self.state = 2
            i, j, value = self.com_graph.comm_swap_edge_disequalize()
            assert(value >= 0)
            if value > 0 and self.com_graph.Lin[i] > 1:
                self.com_graph.dec_intra_edges(i)
                self.com_graph.m += value
                self.com_graph.m = self.com_graph.modularity()  # NEEDED as formula doesn't work
                self.com_graph.inc_intra_edges(j)
                moving = True
        return moving

    def optimize_modularity(self):
        self.state = 0
        running = True
        if self.com_graph.m - self.eps < self.target_m:
            running = False
        while running and abs(self.com_graph.m - self.target_m) > self.eps:
            dec = random.randint(0, 1)
            if dec == 0:
                running = self.increase_modularity1()
            else:
                running = self.increase_modularity2()
                if not running:
                    running = self.increase_modularity1()

        print abs(self.com_graph.m - self.target_m)
        if abs(self.com_graph.m - self.target_m) > self.eps:
            print "Unfeasible"

    def optimize_modularity_legacy(self):
        done = False
        if self.com_graph.m - self.eps < self.target_m:
            # "Unfeasible"
            done = True
        while abs(self.com_graph.m - self.target_m) > self.eps and not done:
            dec = random.randint(0, 1)
            old_state = self.state
            if dec == 0:
                done = not self.replace_internal_external()
            else:
                done = not self.shift_internal()
            if old_state != 0 and self.state != old_state:
                self.state = 0
                done = False
        if done:
            print "Unfeasible"

    def replace_internal_external(self):
        i, j, value = self.com_graph.min_tranf1()
        if self.com_graph.m > self.target_m:  # in state 1
            if self.state == 2 and value >= self.delta_m_cur:
                self.state = 1
                return False
            if self.com_graph.Lin[j] > 0:
                self.com_graph.add_inter_edge(i, j)
                self.delta_m_cur = value
                self.com_graph.m -= self.delta_m_cur
                self.com_graph.dec_intra_edges(j)
                self.state = 1
        else:  # in state 2
            if self.state == 1 and value >= self.delta_m_cur:
                self.state = 2
                return False
            self.delta_m_cur = value
            self.com_graph.m += self.delta_m_cur
            if self.com_graph.num_inter_edges(i, j) > 1:
                self.com_graph.remove_inter_edge(i, j)
                self.com_graph.inc_intra_edges(j)
                self.state = 2
        return True

    def shift_internal(self):
        i, j, value = self.com_graph.min_tranf2()
        if self.com_graph.m > self.target_m:  # state 1
            if self.state == 2 and value >= self.delta_m_cur:
                self.state = 1
                return False
            self.delta_m_cur = value
            self.com_graph.m -= self.delta_m_cur
            self.com_graph.inc_intra_edges(i)
            self.com_graph.dec_intra_edges(j)
            self.state = 1
        else:  # state 2
            if self.state == 1 and value >= self.delta_m_cur:
                self.state = 2
                return False
            self.delta_m_cur = value
            self.com_graph.m += self.delta_m_cur
            self.com_graph.inc_intra_edges(j)
            self.com_graph.dec_intra_edges(i)
            self.state = 2
        return True


def random_graph(G):
    L = len(G.edges())
    n = len(G.nodes())
    part = community.best_partition(G)
    m = community.modularity(part, G)
    c = len(set(part.values()))
    T = TMGG(L, c, m)
    return T.get_random_graph(n)


if __name__ == "__main__":
    import os
    import random
#      G = nx.karate_club_graph()
#      print "modularity: " + \
#          str(community.modularity(community.best_partition(G), G))
#      print "modules: " + str(len(set(community.best_partition(G).values())))
#      H = random_graph(G)
#      print "obtained modularity: " + \
#          str(community.modularity(community.best_partition(H), H))
#      print "Obtained modules: " + str(len(set(community.best_partition(H).values())))
    f = "benchmark_2_1/fortunato/"
    ff = os.listdir(f)
    random.shuffle(ff)
    for l in ff:
        print l
        G = nx.read_edgelist(f + l)
        print "loaded"
        print "modularity: " + \
            str(community.modularity(community.best_partition(G), G))
        print "modules: " + str(len(set(community.best_partition(G).values())))
        H = random_graph(G)
        print "obtained modularity: " + \
            str(community.modularity(community.best_partition(H), H))
        print "Obtained modules: " + str(len(set(community.best_partition(H).values())))
