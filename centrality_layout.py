import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import bsm
import tool
import math
import os
from scipy import stats
import random
import time
import operator



def node_reduce(graph, rate):
    BC = nx.betweenness_centrality(graph)
    # BC = nx.closeness_centrality(graph)
    nodelist = []
    for k in sorted(BC, key=BC.__getitem__, reverse=True):
        nodelist.append(k)

    n = graph.number_of_nodes()
    sub_num = math.ceil(n*rate)
    l = nodelist[0:sub_num]
    sub = nx.subgraph(graph, l)
    print(nx.is_connected(sub))
    return sub


def edge_reduce(graph, rate):
    # graph = node_reduce(graph, 0.8)
    EBC = nx.edge_betweenness_centrality(graph)
    edgelist = []
    for k in sorted(EBC, key=EBC.__getitem__, reverse=True):
        edgelist.append(k)
        graph[k[0]][k[1]]['weight'] = EBC[k]

    # n = graph.number_of_edges()
    # sub_edge_num = math.ceil(n*rate)
    # e = edgelist[n-sub_edge_num:n]
    # sub = nx.edge_subgraph(graph, e)

    # sub = nx.maximum_spanning_tree(graph, weight='weight')

    sub = nx.minimum_spanning_tree(graph, weight='weight')

    print(sub.number_of_edges())
    reinset_sub = sub  # reinsert(sub, edgelist)
    print(reinset_sub.number_of_edges())
    print(nx.is_connected(reinset_sub))
    # plt.subplot(121)
    # pos1 = nx.kamada_kawai_layout(sub, weight=None)
    # nx.draw(sub, pos=pos1,
    #         node_size=1, edge_color='gray')
    return reinset_sub

def eDist(pos, edgelist):
    dist = dict()
    maxD = 0
    for e in edgelist:
        s = e[0]
        t = e[1]
        d = ((pos[s][0]-pos[t][0]) ** 2 + (pos[s][1]-pos[t][1]) ** 2) ** 0.5
        if maxD < d:
            maxD = d
        dist.update({e: d})
    return dist, maxD

def gDist(sub, edgelist):
    dist = dict()
    maxD = 0
    for e in edgelist:
        s = e[0]
        t = e[1]
        d = nx.shortest_path_length(sub, s, t)
        if maxD < d:
            maxD = d
        dist.update({e: d})
    return dist, maxD


def insert(sub, graph, pos, rate_distance, rate_edge_number):
    res = sub
    edgelist = graph.edges()
    print("inserting...")
    dist, maxD = eDist(pos,edgelist)
    # dist, maxD = gDist(sub, edgelist)
    threshold = rate_distance * maxD
    backup = []
    backup_p = []
    print(threshold)
    for k in sorted(dist.items(), key=operator.itemgetter(1)):
        if k[0] not in sub.edges():
            if k[1] < threshold:
                # print(k[1])
                # sub.add_edge(k[0][0], k[0][1])
                backup.append({'s': k[0][0], 't': k[0][1]})
                backup_p.append(1 - graph[k[0][0]][k[0][1]]['weight'])
                # n = n + 1
                # if n > 50:
                # return sub
                # break
            else:
                break
    i = 0
    # n_edge = int(graph.number_of_edges() * rate_edge_number)
    n_edge = int(backup.__len__() * rate_edge_number)
    choose = []
    out_of_range = False
    if n_edge > backup.__len__():
        print('out')
        n_edge = backup.__len__()
        out_of_range = True
    if n_edge > 0:
        random.shuffle(backup)
        choose = backup[0:n_edge-1]
    # while i < n_edge:
    #     # e = random.choices(backup, weights=backup_p)[0]
    #     e = random.choices(backup)[0]
    #     if e in choose:
    #         continue
    #     else:
    #         choose.append(e)
    #         i = i + 1
    #         index = backup.index(e)
    #         backup.pop(index)
    #         backup_p.pop(index)
    for e in choose:
        res.add_edge(e['s'], e['t'], weight=graph[e['s']][e['t']]['weight'])
    print(res.number_of_edges())
    return res, out_of_range


def layout_spring(sub_edge, graph, rate_distance, rate_edge_number, first=None):
    s = nx.Graph(sub_edge)
    time_first = 0.0
    time_insert = 0.0
    time_second = 0.0
    if first == None:
        print("start first spring layout...")
        t = time.time()
        pos_spring = nx.spring_layout(s, weight=None, iterations=300)
        time_first = time.time() - t
    else:
        print("start read first spring layout...")
        pos_spring = first
    t = time.time()
    sub_in, out_of_range = insert(s, graph, pos_spring, rate_distance, rate_edge_number)
    time_insert = time.time() - t
    print("start second spring layout...")
    t = time.time()
    pos_spring_in = nx.spring_layout(
        sub_in, pos=pos_spring, weight=None, iterations=50)
    time_second = time.time() - t
    return sub_in, pos_spring, pos_spring_in, time_first, time_insert, time_second, out_of_range


def layout_bsm(sub_edge, graph, rate_distance, rate_edge_number, first=None):
    s = nx.Graph(sub_edge)
    time_first = 0.0
    time_insert = 0.0
    time_second = 0.0
    if first == None:
        print("start first bsm layout...")
        t = time.time()
        pos_bsm = bsm.solve_bsm(s, 300)
        time_first = time.time() - t
    else:
        print("start read first bsm layout...")
        pos_bsm = first
    t = time.time()
    sub_in, out_of_range = insert(s, graph, pos_bsm, rate_distance, rate_edge_number)
    time_insert = time.time() - t
    print("start second bsm layout...")
    t = time.time()
    # pos_bsm_in = nx.spring_layout(
        # sub_in, pos=pos_bsm, weight=None, iterations=25)
    pos_bsm_in = bsm.solve_bsm(sub_in, 50, pos_bsm, 0)
    time_second = time.time() - t
    return sub_in, pos_bsm, pos_bsm_in, time_first, time_insert, time_second, out_of_range


def layout(graph, color, rate_distance, rate_edge_number):
    # sub_node = graph  # node_reduce(graph, 0.5)
    # print(nx.average_clustering(graph))
    print("start edge filter...")
    sub_edge = edge_reduce(graph, 0.05)
    # print("start spring layout...")
    # pos_spring = nx.spring_layout(sub_edge, weight=None, iterations=200)
    print("start bsm layout...")
    pos_bsm = bsm.solve_bsm(sub_edge, 300)
    sub_in, _ = insert(sub_edge, graph, pos_bsm, rate_distance, rate_edge_number)
    print(nx.average_clustering(sub_in))
    print("start bsm layout...")
    pos_bsm_in = bsm.solve_bsm(sub_in, 50, pos_bsm, 0.2)
    # pos_bsm_in = nx.kamada_kawai_layout(sub_in, pos=pos_bsm)
    # print("start spring layout...")
    # pos_spring_in = nx.spring_layout(
    #     sub_in, pos=pos_spring, weight=None, iterations=100)
    # verify(graph, sub_in)
    # draw(sub_in, pos_spring_in, color)
    draw(sub_in, pos_bsm_in, color)
    return pos_bsm


def draw(g1, pos1, color):
    print("start drawing...")

    plt.figure()
    nx.draw(g1, pos=pos1, node_color=color, nodelist=g1.nodes(),
            node_size=10, edge_color='gray', edge_size=1)
    # nx.draw_networkx_edges(g1, pos=pos1,  # edgelist=print_edges,
    #                        node_size=1, edge_color='black')

    # plt.figure()
    # nx.draw(g2, pos=pos2, node_color=color, nodelist=g2.nodes(),
    #         node_size=10, edge_color='gray', edge_size=1)
    # nx.draw(sub_node, pos=pos_bsm, node_color='r', nodelist=nodelist[0:100],
    # # node_size=10, edge_color='gray', edge_size=1)
    # nx.draw_networkx_edges(g2, pos=pos2,  # edgelist=print_edges,
    #                        node_size=1, edge_color='black')
    # nx.draw_networkx_edges(sub_node, pos=pos_bsm,  fixed=fixed_nodes, edgelist=max_edge,
    #    node_size=1, edge_color='r')


def reinsert(sub, edgelist):
    print("reinserting...")
    for i in range(0, edgelist.__len__()):
        e = edgelist[i]
        next = edgelist[i+1]
        if e not in sub.edges():
            sub.add_edge(e[0], e[1])
        # if sub.number_of_edges() > 500:
        #     return sub
        sub.add_edge(next[0], next[1])
        p, _ = nx.check_planarity(sub)
        sub.remove_edge(next[0], next[1])
        if not p:
            return sub
    return sub


def verify(graph, sub):
    close_g = nx.closeness_centrality(graph)
    close_s = nx.closeness_centrality(sub)
    g = []
    s = []
    for k in sorted(close_g, key=close_g.__getitem__):
        g.append(close_g[k])
    for k in sorted(close_s, key=close_s.__getitem__):
        s.append(close_s[k])
    # print(list(close_g))
    # print(list(close_g))
    print(stats.ks_2samp(normalize(list(g)),
                         normalize(list(s))))


def normalize(x):
    max = np.max(x)
    min = np.min(x)
    a = (x-min)/(max-min)
    return a


if __name__ == "__main__":
    random.seed(0)

    # dir = "./data/facebook_combined"
    dir = "./data/facebook100/Caltech36"
    # dir = "./data/vis"

    graph = tool.readGraph(dir)
    graph = nx.subgraph(graph, max(nx.connected_components(graph), key=len))
    color, _ = tool.render_csv(dir, graph)
    # color2 = tool.render(dir, graph)
    print("node:", graph.number_of_nodes())
    print("edges:", graph.number_of_edges())
    pos = layout(graph, color, 0.15, 0.25)
    # plt.subplot(121)
    # nx.draw(graph, node_color=color,  # with_labels=True, font_size=15,
    # node_size=10, edge_color='gray')
    # pos = nx.spring_layout(graph)
    # nx.draw_networkx_nodes(sub, pos=pos, nodelist=nodelist[-50: -1],
    #                        node_size=1, edge_color='gray')
    # nx.draw_networkx_edges(sub, pos=pos,  edgelist=edgelist[-60:-1],
    #                        node_size=1, edge_color='gray')
    print('Done.')
    plt.show()
