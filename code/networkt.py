import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
import time
import math
from config import *
from numba import jit, njit

test_list = np.array((3, 2, 4, 1, 5, 85), dtype=np.int64)
test_n = 400
test_m = 4
test_gamma = 2.5


def to_full_numpy_matrix(NOCs):
    mat = np.zeros((test_n, test_n))
    for i in NOCs.edges:
        mat[i[0], i[1]] = 1
        mat[i[1], i[0]] = 1
    return mat


@njit
def random_pick_one(p):
    choose_array = []
    for index in range(len(p)):
        for i in range(p[index]):
            choose_array.append(index)
    choice_node = np.random.choice(np.array(choose_array))
    return choice_node


@njit
def random_pick_many(size, p):  # 根据权重p，抽取size个元素，返回下标
    picklist = []
    prob = p.copy()
    for i in range(size):
        picked = random_pick_one(prob)
        picklist.append(picked)
        prob[picked] = 0
    return np.array(picklist)


@njit
def barabasi_albert_graph(N, m, m0):  # BA模型无标度网络
    if m0 is None:
        m0 = m
    BAmat = np.zeros((N, N), dtype=np.int64)
    m0list = np.zeros(1, dtype=np.int64)
    degrees = np.zeros(N, dtype=np.int64)
    for i in range(1, m0):  # 前m0个节点随机连接m0-1条边
        friend = np.random.choice(m0list)
        BAmat[i, friend] = BAmat[friend, i] = 1
        m0list = np.append(m0list, i)
    for i in range(m0):
        degrees[i] = np.count_nonzero(BAmat[i])
    for i in range(m0, N):  # 后来添加的节点需要满足生长和优先依附原则
        choice_node = random_pick_many(m, degrees[:i])
        for j in choice_node:
            BAmat[i, j] = BAmat[j, i] = 1
        for j in range(i+1):
            degrees[j] = np.count_nonzero(BAmat[j])
    return BAmat


# 可以生成连通图
def static_model_scale_free_graph(N, m, gamma):
    SMSFmat = np.zeros((N, N), dtype=np.int64)
    p = np.zeros(N)  # 权重
    sumP = 0.
    for i in range(N):
        # 静态模型的参数 a in [0,1) gamma=(1+a)/a, a=1/(gamma-1)
        p[i] = pow(i+1, -(1/(gamma-1)))  # 第1个节点的编号是0
        sumP += p[i]
    p /= sumP
    edgeNum = 0
    while edgeNum != m * N:
        choice_node = np.random.choice(range(N), size=2, replace=False, p=p)
        if SMSFmat[choice_node[0], choice_node[1]] == 0:
            SMSFmat[choice_node[0], choice_node[1]] = 1
            SMSFmat[choice_node[1], choice_node[0]] = 1
            edgeNum += 1

    return SMSFmat

def connect_static_model_scale_free_graph(N, m, gamma):  # m越大，a越小，越有利于形成单个连通分量
    """
    静态模型生成的连通的无标度网络
    :param N: 网络大小
    :param m: 平均度为2m
    :param gamma: 幂律指数
    :return: networkx中的graph
    静态模型的平均度为2m
    """
    CSMSF_NOCs = nx.empty_graph()
    IsConnect = False
    while not IsConnect:
        CSMSF_NOCs = nx.from_numpy_array(static_model_scale_free_graph(N, m, gamma))
        IsConnect = nx.is_connected(CSMSF_NOCs)
        print(nx.number_connected_components(CSMSF_NOCs))
        # for i in nx.connected_components(CSMSF_NOCs):
        #     print(i)
    return CSMSF_NOCs


def qtt_static_model_scale_free_graph(N, k_av, a):  # 连通图static model模型，qtt改造版
    pp = np.zeros((1, N), dtype=np.int64)
    A = np.zeros(N, dtype=np.int64)
    number_of_edges = k_av/2*N
    for i in range(N):
        pp[i] = i ^ (-a)
    pp = pp/sum(pp)
    p = pp
    for i in range(2, N):
        p[i] = sum(pp[1:i-1]) + p[i]
    edges = 0
    # while edges < number_of_edges:
    #     t1 = rand();
    #     node1 = find(p > t1, 1);
    #
    #     t2 = rand();
    #     node2 = find(p > t2, 1);
    #
    #     if A[node1, node2] == 0 and node1 != node2:
    #         edges = edges + 1
    #         A[node1, node2] = 1
    #         A[node2, node1] = 1
    return A

@njit
def erdos_renyi_graph(N, p):  # ER随机图，N个节点
    ERmat = np.zeros((N, N), dtype=np.int64)
    for i in range(N):
        for j in range(i):
            if p > random.random():
                ERmat[i, j] = ERmat[j, i] = 1
    return ERmat


def HK_SF_graph(N, m, m0, p):  # HK模型，可以调整网络聚集系数
    """

    :param N:
    :param m:
    :param m0:
    :param p:
    :return: 一个nx类型网络
    """

    G = nx.gnp_random_graph(m0, 0.05)
    print(nx.is_connected(G))
    # G = nx.Graph()
    # G.add_nodes_from(range(m0))  # 初始的m0个节点
    # for i in G.nodes():  # 初始连接m0-1条边，保证图连通
    #     j = i
    #     while j == i or G.degree[j] != 0:
    #         j = np.random.choice(a=range(m0), size=1)
    #     G.add_edge(i, j)

    return G


if __name__ == '__main__':
    CSMSF_NOC = connect_static_model_scale_free_graph(test_n, test_m, test_gamma)  # m越大，a越小，越有利于形成单个连通分量
    #
    tempMat = [to_full_numpy_matrix(CSMSF_NOC)]
    np.save("./snapshot_CSMSF/CSMSF_" + str(test_n) + "N_" + str(test_m) + "m_" + str(test_gamma) + "gamma" + ".npy", np.array(tempMat))
    # nx.write_gexf(CSMSF_NOC, "./draw_gexf/static_CSMSF.gexf")
    # HK_SF_graph(test_n, test_m, 10, 0.3)


    # sum_deg = np.zeros(test_n, dtype=np.int64)
    # degree = np.zeros(test_n, dtype=np.int64)
    # for i in range(test_n):
    #     degree[i] = np.count_nonzero(CSMSF_NOC[i])
    # for i in range(test_n):
    #     sum_deg[degree[i]] += 1
    # print(sum_deg)
    # xpoint = []
    # ypoint = []
    # for i in range(1, test_n):
    #     if sum_deg[i] != 0:
    #         xpoint.append(math.log(i))
    #         ypoint.append(math.log(sum_deg[i]))
    #
    # # print(xpoint)
    # # print(ypoint)
    # plt.figure()
    # plt.scatter(xpoint, ypoint)
    # plt.show()




