import random
import numpy as nm
import networkx as nx
import scipy.stats.stats as stats
import scipy.sparse.csgraph as csg
import community


def get_joint_degrees(A):
    joint_degrees = {}

    n = nm.shape(A)[0]
    u = nm.ones((1, n))
    N = nm.ravel(u*A)
    for i in N:
        joint_degrees[int(i)] = {}

    B = A*nm.diag(N)
    for i in range(n):
        for j in range(n):
            if B[i, j]:
                if B[j, i] in joint_degrees[B[i, j]]:
                    joint_degrees[int(B[i, j])][int(B[j, i])] += 1
                else:
                    joint_degrees[int(B[i, j])][int(B[j, i])] = 1
            j += 1
        i += 1
    return joint_degrees


def is_connected(A):
    n = nm.shape(A)[0]
    u = nm.zeros((n, 1))
    u[0, 0] = 1
    connected = True

    for i in range(n + 1):
        u = u + A * u
        u /= nm.sum(u)

    i = 0
    while (i < n and connected):
        if u[i, 0] == 0:
            connected = False
        i += 1

    return connected


def average_clustering(A):
    n = nm.shape(A)[0]
    neighs = A*nm.ones((n, 1))
    c = 0.0
    tris = A**3
    for i in range(n):
        if neighs[i] > 1:
            c += float(tris[i, i]/(neighs[i] * (neighs[i] - 1)))
    return c/n


def clustering(A):
    n = nm.shape(A)[0]
    neighs = A*nm.ones((n, 1))
    c = {}
    tris = A**3
    for i in range(n):
        if neighs[i] > 1:
            c[i] = float(tris[i, i]/(neighs[i] * (neighs[i] - 1)))
    return c


def average_clustering_old(A):
    n = nm.shape(A)[0]
    C = []
    for i in range(n):
        neig = []
        tris = 0
        for j in range(n):
            if A[j, i] != 0:
                for k in neig:
                    if A[k, j] != 0:
                        tris += 1
                neig.append(j)
        if len(neig) > 1:
            C.append(2.0*tris/(len(neig)*(len(neig)-1)))

    if len(C):
        return sum(C)/n
    else:
        return 0


def get_degrees(A):
    u = nm.ones((1, nm.shape(A)[0]))
    return nm.sort(nm.ravel(u*A))


def num_edges(A):
    return sum(get_degrees(A))/2


def get_avg_neighbour_degree(A):
    degs = get_joint_degrees(A)
    n = int(max(degs))

    annd = {}  # nm.ravel(nm.zeros((1, n)))
    for k in range(1, n + 1):
        if k in degs:
            annd[k] = sum([v*k for v in degs[k].values()]) /\
                sum(degs[k].values())
    return annd


def get_common_neigh_dist(A):
    cnd = {}
    n = nm.shape(A)[0]

    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] != 0:
                v = float(nm.transpose(A[:, i]) * (A[:, j]))
                if v > 0:
                    if v in cnd:
                        cnd[v] += 1
                    else:
                        cnd[v] = 1.
    for m in cnd:
        cnd[m] /= n * (n - 1) / 2.
    return cnd


def get_kcoreness(B):
    A = B.copy()
    n = nm.shape(A)[0]
    kcore = {1: n}
    u = nm.ones((1, n))

    deg = nm.ravel(u * A)
    for v in range(2, int(max(deg)) + 1):
        pruning = True
        while pruning:
            pruning = False
            for i in range(n):
                if deg[i] > 0 and deg[i] < v:
                    A[:, i] = nm.zeros((n, 1))
                    A[i, :] = nm.zeros((1, n))
                    pruning = True
            deg = nm.ravel(u * A)
            kcore[v] = 0
            for i in range(n):
                if deg[i] > 0:
                    kcore[v] += 1
    return kcore


def get_degree_betweeness(A):
    degbet = {}
    n = nm.shape(A)[0]
    G = nx.from_numpy_matrix(A)
    degs = nx.degree(G)
    bws = nx.betweenness_centrality(G)

    for d in range(1, int(max(degs.values())) + 1):
        degbet[d] = 0
        s = 0
        for i in range(n):
            if degs[i] == d:
                degbet[d] += bws[i]
                s += 1
        if degbet[d] > 0:
            degbet[d] /= s

    return degbet


def correlation(d1, d2):
    data1 = nm.ravel(d1).copy()
    data2 = nm.ravel(d2).copy()
    le = max(len(data1), len(data2))
    data1.resize((le,))
    data2.resize((le,))
    return stats.pearsonr(data1, data2)[0]


def all_pairs_shortest_path_length(A):
    n = nm.shape(A)[0]
    B = nm.eye(n)
    path = {}

    for k in range(n):
        B = B * A
        for i in range(n):
            for j in range(i + 1, n):
                if B[i, j] and ((i, j) not in path):
                    path[(i, j)] = k + 1
    return path


def node_distance_dist(A):
    n = nm.shape(A)[0]
    path_lengths = csg.shortest_path(A, unweighted=True)

    dist = {}
    for i in range(n):
        for j in range(i + 1, n):
            k = path_lengths[i, j]
            if k <= n:
                if k in dist:
                    dist[k] += 1
                else:
                    dist[k] = 1.

    for k in dist:
        dist[k] /= n * (n - 1) / 2.

    return dist


def node_distance_dist_old(A):
    n = nm.shape(A)[0]
    G = nx.from_numpy_matrix(A)
    path_lengths = nx.all_pairs_shortest_path_length(G)

    dist = {}
    for i in range(n):
        for j in range(i + 1, n):
            if j in path_lengths[i]:
                k = path_lengths[i][j]
                if k in dist:
                    dist[k] += 1
                else:
                    dist[k] = 1.

    for k in dist:
        dist[k] /= n * (n - 1) / 2.

    return dist


def correlate_dist_dict(d1, d2):
    if len(d1) > 0 and len(d2) == 0:
        return 0
    if len(d1) == 0 and len(d2) > 0:
        return 0
    if len(d1) == 0 and len(d2) == 0:
        return 1

    minv = min(min(d1), min(d2))
    maxv = max(max(d1), max(d2))
    n = int(maxv - minv + 1)
    v1 = nm.zeros((n, 1))
    v2 = nm.zeros((n, 1))

    for v in d1:
        v1[int(v - minv)] = d1[v]
    for v in d2:
        v2[int(v - minv)] = d2[v]

    return correlation(v1, v2)


def sorted_correlation(l1, l2):
    n = max(len(l1), len(l2))
    v1 = nm.zeros((n, 1))
    v2 = nm.zeros((n, 1))

    el1 = nm.argsort(l1)
    el2 = nm.argsort(l2)
    for i in range(len(el1)):
        v1[i] = l1[i]
    for i in range(len(el2)):
        v2[i] = l2[i]

    return correlation(v1, v2)


def simm_matrix_2_graph(A, names=None):
    n = nm.shape(A)[0]
    G = nx.Graph()
    for i in range(n):
        if names:
            G.add_node(names[i])
        else:
            G.add_node(i)
        for j in range(i, n):
            if A[i, j]:
                if names:
                    G.add_edge(names[i], names[j])
                else:
                    G.add_edge(i, j)
    return G


def inv_map(d):
    i = {}
    for k, v in d.iteritems():
        if v in i:
            i[v] += [k]
        else:
            i[v] = [k]
    return i


def correlate_modularity(G, H):
    g = community.best_partition(G)
    h = community.best_partition(H)
    g = inv_map(g)
    h = inv_map(h)
    g = {k: len(v) for k, v in g.iteritems()}
    h = {k: len(v) for k, v in h.iteritems()}
    return correlate_dist_dict(g, h)


def ilogit(x):
    ''' inverse of the logit function '''
    return nm.exp(x) / (1 + nm.exp(x))


def logistic(x):
    return 1. / (1 + nm.exp((0.5-x)*6))


def truncate(x):
    if x < 0:
        return 0
    if x > 1:
        return 1
    return x


def mateify(B):
    n = nm.shape(B)[0]
    one = nm.ones((1, n))
    neighs = nm.ravel(one * B)
    for i in range(n):
        if neighs[i] == 0:
            j = random.randint(0, n - 1)
            B[i, j] = 1
            B[j, i] = 1


def connect_components(H):
    groups = sorted(nx.connected_components(H), key=len, reverse=True)

    while len(groups) > 1:
        w = {i: len(groups[i]) for i in range(len(groups))}
        k = random.randint(0, len(groups) - 1)
        r = dict(w)
        del(r[k])
        tot_prob = float(sum(r.values()))
        c = nm.random.choice(r.keys(), 1, p=[v/tot_prob for v in r.values()])[0]

        i = random.sample(groups[k], 1)[0]
        j = random.sample(groups[c], 1)[0]
        H.add_edge(i, j)

        groups[k] = groups[k].union(groups[c])
        del groups[c]
        w[k] += w[c]
        del w[c]


def norm_modularity(G, part=None):
    if not part:
        part = community.best_partition(G)
    degs = G.degree()
    m = len(G.edges())

    mod_map = {}
    for k, v in part.iteritems():
        mod_map[v] = mod_map.get(v, [])
        mod_map[v].append(k)

    s = 0
    intra_edge = 0
    for v, nodes in mod_map.iteritems():
        for i in range(len(nodes)-1):
            for j in range(i+1, len(nodes)):
                s += degs[nodes[i]]*degs[nodes[j]]/(2.*m)
                if G[nodes[i]].get(nodes[j], None) is not None:
                    intra_edge += 1

    return (intra_edge - s)/(m - s), len(mod_map.keys())


def community_size(G):
    part = community.best_partition(G)
    sizes = {}
    for k in part:
        sizes[part[k]] = sizes.get(part[k], 0) + 1
    return sizes.values()
