import random
import networkx as nx
import multiprocessing
from player import playersInit
from config import *
from functools import partial
from math import exp
from datetime import datetime


def get_name(graphtype, f, subNet):
    params_dict = {
        'SF': str(m) + "m",
        'ER': str(ER_p) + "p",
        'CSMSF': str(m) + "m_" + str(gamma) + "gamma",
        'RR': str(d) + "d"
    }

    name = f"{graphtype}_{N}N_{params_dict[graphtype]}_{f}f_{subNet}"
    return name


def readSnapshot(graph_type, f, sub_net, netk=m, model='Original', path='.'):
    filename = path + f"/{snapshot_name}/" + get_name(graph_type, f, sub_net)

    if model != 'Original':
        filename += f"_{model}"

    filename += ".npy"
    print(filename)
    return np.load(filename, allow_pickle=True)


def play(x, y, _PayOff):  # 双方进行博弈，返回收益  x,y为index
    if players[x].strategy and players[y].strategy:  # 都合作
        return _PayOff.R
    elif players[x].strategy and not players[y].strategy:
        return _PayOff.S
    elif not players[x].strategy and players[y].strategy:
        return _PayOff.T
    else:  # 都对抗
        return _PayOff.P


def fermi_strategyUpdate(x, y):  # 策略的更新过程
    try:
        prob = 1 / (1 + exp(-1 * s * (players[y].AccPayOffs - players[x].AccPayOffs)))
    except OverflowError:
        prob = 1 if players[y].AccPayOffs > players[x].AccPayOffs else 0

    if prob > random.random():
        players[x].newStrategy = players[y].strategy
        return
    players[x].newStrategy = players[x].strategy


def EvolutionGameStep(NOCs, bORr):  # 一轮演化过程
    _PayOff = PayOff_PD(bORr)

    # 博弈收益
    for _id in NOCs.nodes():
        if NOCs.degree[_id] != 0:
            for friend in NOCs.adj[_id]:
                players[_id].AccPayOffs += play(_id, friend, _PayOff)

    # 策略更新
    for _id in NOCs.nodes():
        if NOCs.degree[_id] != 0:
            friend = random.choice(list(NOCs.adj[_id]))
            fermi_strategyUpdate(_id, friend)

    Temp = 0.
    for _id in NOCs.nodes():
        players[_id].AccPayOffs = 0  # 每轮都要清零一次
        players[_id].strategy = players[_id].newStrategy
        if players[_id].strategy:
            Temp += 1
    return Temp / N


def EveryRoundEvolutionGame(round, _tempGraph, _g, _bORr):  # 图上的一个点的每个round
    fc_net = 0.
    playersInit()
    tempGraphIndex = 0
    random.seed()  # 多进程内部利用时间进行随机数重置
    for step in range(G1 + G2):
        if step % _g == 0:
            tempGraphIndex = (step // _g) % snapshotNum

        fc = EvolutionGameStep(nx.from_numpy_array(_tempGraph[tempGraphIndex]), _bORr)

        if fc == 0 or fc == 1:  # 一方灭绝，提前终止
            if step >= G1:
                fc_net += (G1 + G2 - step - 1) * fc
                fc_net /= G2
                print("Round:{}, step:{}, b:{}, fc:{}".format(round, step, _bORr, fc_net))
                return fc_net
            print("Round:{}, step:{}, b:{}, fc:{}".format(round, step, _bORr, fc))
            return fc

        if step >= G1:
            fc_net += fc
    fc_net /= G2
    print("Round:{}, b:{}, fc:{}".format(round, _bORr, fc_net))
    return fc_net


def EvolutionGameProcess(net, subnet, g, f, netk=m, model='Original'):  # 跑图上的一条线
    tempGraph = readSnapshot(net, f, subnet, netk, model)
    yPoint = []
    num_processes = multiprocessing.cpu_count() - 2  # 使用核心数
    for b in blist:
        func = partial(EveryRoundEvolutionGame, _tempGraph=tempGraph, _g=g, _bORr=b)
        pool = multiprocessing.Pool(processes=num_processes)
        yi = sum(list(pool.imap(func, range(EG_Rounds)))) / EG_Rounds
        print("***************************************g:{}, f:{}, b:{}, fc:{}".format(g, f, b, yi))
        pool.close()
        pool.join()
        yPoint.append(yi)

    with open('./' + result_name + '/' + get_name(net, f, subnet) + str(g) + 'g_' + model + '.txt', 'w') as f:
        for i in range(len(yPoint)):
            f.write(str(blist[i]) + ' ' + str(yPoint[i]) + '\n')
