import numpy as np
players = []
# seed_value = 2022
snapshot_name = 'snapshot'
result_name = 'result'

# Network parameters
N = 400  # Network size
graphType = ['SF']
# subNet = ['Static', 'Random', 'Single-cluster', 'Multi-cluster', 'Single-star', 'Multi-star']
subNet = ['Random', 'Single-cluster', 'Multi-cluster', 'Single-star', 'Multi-star']

snapshotNum = 200
m = 4  # Parameter for BA model
gamma = 2.3  # Parameter for static model
ER_p = 0.01  # Parameter for ER graph
d = 8  # Parameter for RR graph
s = 2  # Selection intensity for the Fermi function, strong selection

# Evolutionary parameters
G1 = 50000  # Pre-evolution steps
G2 = 2000  # Averaging steps

DiffGraph = 1  # Number of different graphs to average over
EG_Rounds = 100  # Number of game rounds per graph
b_n = 16
blist = np.linspace(1.0, 2.5, b_n)

glist = [100, 500]
flist = [0.3]

# Control flags
flag_save = True  # Whether to save the current snapshot
flag_draw = 3  # 1: Draw with matplotlib, 2: Draw with Gephi, 3: Do not draw
flag_complex = False  # Whether to output detailed information for each snapshot
flag_static = True  # Whether to recreate the static network
flag_change_snapshot = False  # Whether to modify the generated temporal network

# Game payoff settings
class PayOff_PD:  # Payoffs for the Prisoner's Dilemma
    def __init__(self, b):
        self.T = b
        self.R = 1
        self.P = 0
        self.S = 0

class PayOff_SG:  # Payoffs for the Snowdrift Game
    def __init__(self, r=0.5):
        beta = (1 / r + 1)/2
        self.T = beta
        self.R = beta - 0.5
        self.P = 0
        self.S = beta - 1
