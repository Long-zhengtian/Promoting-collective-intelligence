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


def play(x, y, _PayOff):  # Interaction between two players, return payoff; x, y are indices
    if players[x].strategy and players[y].strategy:  # Both cooperate
        return _PayOff.R
    elif players[x].strategy and not players[y].strategy:
        return _PayOff.S
    elif not players[x].strategy and players[y].strategy:
        return _PayOff.T
    else:  # Both defect
        return _PayOff.P


def fermi_strategyUpdate(x, y):  # Strategy update based on Fermi rule
    try:
        prob = 1 / (1 + exp(-1 * s * (players[y].AccPayOffs - players[x].AccPayOffs)))
    except OverflowError:
        prob = 1 if players[y].AccPayOffs > players[x].AccPayOffs else 0

    if prob > random.random():
        players[x].newStrategy = players[y].strategy
        return
    players[x].newStrategy = players[x].strategy


def EvolutionGameStep(NOCs, bORr):  # A single round of evolution
    _PayOff = PayOff_PD(bORr)

    # Game payoff
    for _id in NOCs.nodes():
        if NOCs.degree[_id] != 0:
            for friend in NOCs.adj[_id]:
                players[_id].AccPayOffs += play(_id, friend, _PayOff)

    # Strategy update
    for _id in NOCs.nodes():
        if NOCs.degree[_id] != 0:
            friend = random.choice(list(NOCs.adj[_id]))
            fermi_strategyUpdate(_id, friend)

    Temp = 0.
    for _id in NOCs.nodes():
        players[_id].AccPayOffs = 0  # Reset payoff after each round
        players[_id].strategy = players[_id].newStrategy
        if players[_id].strategy:
            Temp += 1
    return Temp / N


def EveryRoundEvolutionGame(round, _tempGraph, _g, _bORr):  # One game instance for each round on the graph
    fc_net = 0.
    playersInit()
    tempGraphIndex = 0
    random.seed()  # Reset RNG in subprocesses using current time
    for step in range(G1 + G2):
        if step % _g == 0:
            tempGraphIndex = (step // _g) % snapshotNum

        fc = EvolutionGameStep(nx.from_numpy_array(_tempGraph[tempGraphIndex]), _bORr)

        if fc == 0 or fc == 1:  # One strategy dominates, early termination
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


def EvolutionGameProcess(net, subnet, g, f, netk=m, model='Original'):  # Main loop for one line in the figure
    tempGraph = readSnapshot(net, f, subnet, netk, model)
    yPoint = []
    num_processes = multiprocessing.cpu_count() - 2  # Number of processes to use
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
