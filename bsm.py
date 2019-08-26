import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from networkx.algorithms import approximation as apxa
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import bicgstab
import math
import tool
import time
import BHTree
from random import uniform
from ctypes import *
from cg.utils import get_solver

class tempPointer(Structure):
        pass

class PyRes(Structure):  
    _fields_ = [("first", c_double), ("second", c_double)]  

def EuclideanDistances(A, B):
    BT = B.transpose()
    vecProd = A * BT
    SqA = A.getA()**2
    sumSqA = np.matrix(np.sum(SqA, axis=1))
    sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))
    SqB = B.getA()**2
    sumSqB = np.sum(SqB, axis=1)
    sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))
    SqED = sumSqBEx + sumSqAEx - 2*vecProd
    SqED[SqED < 0] = 0
    ED = (SqED.getA())**0.5
    return np.matrix(ED)


def normalize(x):
    max = np.max(x)
    min = np.min(x)
    a = 2*((x-min)/(max-min))-1
    return a


def read_txt(graphDir):
    name = graphDir.split("/")
    graphFile = graphDir + "/" + name[-1] + ".txt"
    g = nx.read_edgelist(graphFile, create_using=nx.Graph(), nodetype=int)
    return g

def mcg(A, b, x):
    r = b - np.matmul(A, x) 
    p = np.copy(r) 
    i = 0 
    while(max(abs(r)) > 1e-5 and i < 1000): 
        # print('i',i) 
        # print('r',r) 
        pap = np.dot(np.dot(A, p), p) 
        if pap == 0: #分母太小时跳出循环 
            return x 
        # print('pap=',pap) 
        alpha = np.dot(r, r) / pap #直接套用公式 
        x1 = x + alpha * p 
        r1 = r - alpha * np.dot(A, p) 
        beta = np.dot(r1, r1) / np.dot(r, r) 
        p1 = r1 + beta * p 
        r = r1 
        x = x1 
        p = p1 
        i = i + 1 
    return x, i


def solve(graph, matrixM, matrixL, pos, c):
    M = matrixM
    L = matrixL
    n = graph.number_of_nodes()
    B = np.zeros((n, 2))
    
    t = time.time()
    ED = EuclideanDistances(np.matrix(pos), np.matrix(pos))  # 欧氏距离
    ED = ED + np.eye(n)
    chaX = np.tile(pos[:, 0], [n, 1]) - \
        np.tile(np.matrix(pos[:, 0]).transpose(), [1, n])  # 坐标差
    Btemp = chaX/ED
    B[:, 0] = np.nansum(Btemp, axis=0)
    chaY = np.tile(pos[:, 1], [n, 1]) - \
        np.tile(np.matrix(pos[:, 1]).transpose(), [1, n])
    Btemp = chaY/ED
    B[:, 1] = np.nansum(Btemp, axis=0)
    # print(time.time() - t)

    t = time.time()
    mAndL = M + c*n*L
    pos[:, 0], _ = cg(mAndL, B[:, 0], pos[:, 0])
    pos[:, 1], _ = cg(mAndL, B[:, 1], pos[:, 0])

    pos[:, 0] = normalize(pos[:, 0])
    pos[:, 1] = normalize(pos[:, 1])
    # print(time.time() - t)
    return pos

def calBHTree(n, pos):
    root = BHTree.TreeNode([0, 0], 2)
    pts = []
    t = time.time()
    for i in range(n):
        pts.append(BHTree.Point(i, pos[i]))
        root.insert(pts[i])
    
    tree = BHTree.Tree(root, 1.5)
    B = np.zeros((n, 2))
    print(time.time() - t)
    t = time.time()
    for i in range(n):
        res = tree.sinAndCos(pts[i])
        B[i] = np.array(res)
    print(time.time() - t)
    return B

def calBHTreeC(n, pos):
    mylib = cdll.LoadLibrary("./BHtree.so")

    mylib.setNodeConfig.restype = POINTER(tempPointer)  
    mylib.insert.restype = POINTER(tempPointer)
    mylib.calSinCos.restype = PyRes 

    root = mylib.setNodeConfig(c_double(0),c_double(0),c_double(2))

    pts = []
    t = time.time()

    for i in range(n):
        temp = mylib.insert(c_int(i), c_double(pos[i][0]), c_double(pos[i][1]))
        pts.append(temp)
    # print(time.time() - t)
    B = np.zeros((n, 2))
    t = time.time()
    for i in range(n):
        res = mylib.calSinCos(pts[i], c_double(0.2))
        B[i][0] = res.first
        B[i][1] = res.second

    print(time.time() - t)
    return B

def solve_BH(graph, matrixM, matrixL, pos, c):
    M = matrixM.copy()
    L = matrixL.copy()
    n = graph.number_of_nodes()
    
    B = calBHTreeC(n, np.copy(pos))
    B = normalize(B)

    t = time.time()
    mAndL = M + c*n*L
    mAndL = normalize(mAndL)


    pos[:, 0], a = cg(mAndL, B[:, 0], pos[:, 0])
    pos[:, 1], _ = cg(mAndL, B[:, 1], pos[:, 1])
    print(a)
    # print(np.dot(mAndL, pos[:, 0]) - B[:, 0])
    # pos[:, 0] = normalize(P_x)[0]
    # pos[:, 1] = normalize(P_y)[0]
    pos[:, 0] = normalize(pos[:, 0])
    pos[:, 1] = normalize(pos[:, 1])
    print(time.time() - t)
    return pos

def matrixInit(graph):
    # n = np.max(np.array(graph.nodes()))+1
    n = graph.number_of_nodes()
    node_dict = {}
    i = 0
    for v in graph.nodes():
        node_dict.update({v: i})
        i = i + 1

    matrixM = -1 * np.ones((n, n), dtype=float) + n * np.eye(n)
    # matrixM = n * np.eye(n)
    matrixL = np.zeros((n, n), dtype=float)
    for edge in graph.edges:
        source = node_dict[edge[0]]
        target = node_dict[edge[1]]
        matrixL[source][target] += -1
        matrixL[target][target] += 1
        matrixL[target][source] += -1
        matrixL[source][source] += 1
    # print(matrixL)
    # print(matrixM)
    pos = np.random.uniform(-1, 1, size=(n, 2))
    return matrixM, matrixL, pos, node_dict


def solve_bsm(graph, iter, initPos='none', firstIterRate=0.95):
    matrixM, matrixL, pos, node_dict = matrixInit(graph)
    if initPos != 'none':
        for v in graph.nodes():
            i = node_dict[v]
            pos[i][0] = initPos[v][0]
            pos[i][1] = initPos[v][1]

    iter1 = math.ceil(firstIterRate*iter)
    for i in range(0, iter1):
        # print(i)
        pos = solve(graph, matrixM, matrixL, pos, 1000)
    for i in range(0, iter - iter1):
    # for i in range(0, 1):
            # print(i)
        pos = solve(graph, matrixM, matrixL, pos, 1)
    position = {}
    for v in graph.nodes():
        i = node_dict[v]
        x = pos[i][0]
        y = pos[i][1]
        position.update({v: np.array([x, y])})
    return position


if __name__ == "__main__":
    # graph = read_txt("./data/facebook100/Caltech36")
    # dir = "./data/facebook100/Caltech36"
    dir = "./data/web-edu"
    name = dir.split("/")[-1]
    graph = tool.readGraph(dir)
    # graph = nx.subgraph(graph, max(nx.connected_components(graph), key=len))
    # nodeColor, nodeClassInCD = tool.render_csv(dir, graph)
    position = solve_bsm(graph, 10)
    nx.draw(graph,  pos=position,
            node_size=1, edge_color='gray')
    # plt.figure(figsize=(10, 10))
    # nx.draw(graph,node_color=nodeColor, nodelist=graph.nodes(),
            # node_size=10, edge_color='gray', edge_size=1)
    # nx.draw_networkx_edges(g, pos=pos,  edgelist=sub.edges(),
    #                         edge_color='black')
    # savedir = dir + '/hair.pdf'
    # plt.savefig(savedir, dpi=300)
    # plt.close()
    # BC = nx.degree_centrality(graph)
    # print(max(BC.values()))
    # distance = nx.shortest_path_length(graph, source=0)
    # a = distance[4]
    # plt.savefig("ba.png")           #输出方式1:将图像存为一个png格式的图片文件
    # print(nx.edges(graph))
    plt.show()
