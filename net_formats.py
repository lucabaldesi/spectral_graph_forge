#! /usr/bin/env python

import lxml.etree as et
import networkx as nx
import scipy.io as io


def read_graphml(filename):
    tree = et.parse(filename)
    G = nx.Graph()

    graphml = {
        "graph": "{http://graphml.graphdrawing.org/xmlns}graph",
        "node": "{http://graphml.graphdrawing.org/xmlns}node",
        "edge": "{http://graphml.graphdrawing.org/xmlns}edge",
        "data": "{http://graphml.graphdrawing.org/xmlns}data",
        "label": "{http://graphml.graphdrawing.org/xmlns}data[@key='label']",
        "x": "{http://graphml.graphdrawing.org/xmlns}data[@key='x']",
        "y": "{http://graphml.graphdrawing.org/xmlns}data[@key='y']",
        "size": "{http://graphml.graphdrawing.org/xmlns}data[@key='size']",
        "r": "{http://graphml.graphdrawing.org/xmlns}data[@key='r']",
        "g": "{http://graphml.graphdrawing.org/xmlns}data[@key='g']",
        "b": "{http://graphml.graphdrawing.org/xmlns}data[@key='b']",
        "weight": "{http://graphml.graphdrawing.org/xmlns}data[@key='weight']",
        "edgeid": "{http://graphml.graphdrawing.org/xmlns}data[@key='edgeid']"
    }

    graph = tree.find(graphml.get("graph"))
    edges = graph.findall(graphml.get("edge"))
    edges = [(e.attrib['source'], e.attrib['target']) for e in edges]

    G.add_edges_from(edges)
    return G


def read_mat_file(filename, mat_key='A'):
    mat_cont = io.loadmat(filename, mdict=None, appendmat=True)
    A = mat_cont[mat_key]
    return nx.from_scipy_sparse_matrix(A)

if __name__ == "__main__":
    read_graphml("cat_brain.graphml")
