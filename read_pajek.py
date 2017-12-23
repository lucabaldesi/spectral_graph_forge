#!/usr/bin/env python


import networkx as nx
import shlex


def line_state(line):
    if line.startswith("*Network"):
        return "nodes"
    elif line.startswith("*Vertices"):
        return "filling"
    elif line.startswith("*Arcs"):
        return "edges"
    elif line.startswith("*Partition"):
        temp = "-".join(line.split(" ")[1:]).strip()
        return "".join(temp.split("_")[1:])
    else:
        return None


def read_pajek(filename, edge_attribute=True):
    state = None
    nodes = {}
    G = nx.Graph()
    with open(filename) as f:
        line = f.readline().strip()
        while line:
            new_state = line_state(line)
            if new_state != "filling":
                if new_state:
                    state = new_state
                    count = 1
                else:  # actual data
                    data = shlex.split(line)
                    if state == "nodes":
                        nodes[int(data[0])] = data[1]
                        G.add_node(data[1])
                    elif state == "edges":
                        if edge_attribute:
                            G.add_edge(nodes[int(data[0])],
                                       nodes[int(data[1])],
                                       weight=float(data[2]))
                        else:
                            G.add_edge(nodes[int(data[0])],
                                       nodes[int(data[1])])
                    else:  # node attribute
                        G.node[nodes[count]][state] = data[0]
                    count += 1
            line = f.readline()
    return G


if __name__ == "__main__":
    read_pajek("add_health/comm4.paj")
