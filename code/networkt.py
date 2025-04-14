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
def random_pick_many(size, p):
    picklist = []
    prob = p.copy()
    for i in range(size):
        picked = random_pick_one(prob)
        picklist.append(picked)
        prob[picked] = 0
    return np.array(picklist)


@njit
def barabasi_albert_graph(N, m, m0):
    if m0 is None:
        m0 = m
    BAmat = np.zeros((N, N), dtype=np.int64)
    m0list = np.zeros(1, dtype=np.int64)
    degrees = np.zeros(N, dtype=np.int64)
    for i in range(1, m0):
        friend = np.random.choice(m0list)
        BAmat[i, friend] = BAmat[friend, i] = 1
        m0list = np.append(m0list, i)
    for i in range(m0):
        degrees[i] = np.count_nonzero(BAmat[i])
    for i in range(m0, N):
        choice_node = random_pick_many(m, degrees[:i])
        for j in choice_node:
            BAmat[i, j] = BAmat[j, i] = 1
        for j in range(i+1):
            degrees[j] = np.count_nonzero(BAmat[j])
    return BAmat


def static_model_scale_free_graph(N, m, gamma):
    SMSFmat = np.zeros((N, N), dtype=np.int64)
    p = np.zeros(N)  # 权重
    sumP = 0.
    for i in range(N):
        p[i] = pow(i+1, -(1/(gamma-1)))
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

def connect_static_model_scale_free_graph(N, m, gamma):
    CSMSF_NOCs = nx.empty_graph()
    IsConnect = False
    while not IsConnect:
        CSMSF_NOCs = nx.from_numpy_array(static_model_scale_free_graph(N, m, gamma))
        IsConnect = nx.is_connected(CSMSF_NOCs)
        print(nx.number_connected_components(CSMSF_NOCs))
        # for i in nx.connected_components(CSMSF_NOCs):
        #     print(i)
    return CSMSF_NOCs

@njit
def erdos_renyi_graph(N, p):
    ERmat = np.zeros((N, N), dtype=np.int64)
    for i in range(N):
        for j in range(i):
            if p > random.random():
                ERmat[i, j] = ERmat[j, i] = 1
    return ERmat


if __name__ == '__main__':
    CSMSF_NOC = connect_static_model_scale_free_graph(test_n, test_m, test_gamma)
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




