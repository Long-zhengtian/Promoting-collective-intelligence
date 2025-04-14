import random
import os
import networkx as nx
from config import *
from EvolutionGame import EvolutionGameProcess

# Set random seeds (uncomment if needed)
# random.seed(seed_value)
# np.random.seed(seed_value)
# os.environ['PYTHONHASHSEED'] = str(seed_value)

if __name__ == '__main__':
    # Static network
    for net in graphType:
        EvolutionGameProcess(net, 'Static', G1+G2, 0)

    # Subnetworks (used for temporal network evolution)
    for net in graphType:
        for s in subNet:  # Subnetwork types
            for g in glist:  # Number of evolution rounds per snapshot
                for f in flist:  # Proportion of participating nodes
                    for model in ['Original']:  # Model type
                        EvolutionGameProcess(net, s, g, f, m, model)
