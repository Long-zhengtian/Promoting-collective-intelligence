import random
import os
import networkx as nx
from config import *
from EvolutionGame import EvolutionGameProcess

# random.seed(seed_value)
# np.random.seed(seed_value)
# os.environ['PYTHONHASHSEED'] = str(seed_value)

if __name__ == '__main__':
    # static network
    for net in graphType:
        EvolutionGameProcess(net, 'Static', G1+G2, 0)

    # 子网络
    for net in graphType:
        for s in subNet:
            for g in glist:  # 每个snapshot的演化次数
                for f in flist:  # 节点的比例
                    for model in ['Original']:
                        EvolutionGameProcess(net, s, g, f, m, model)



