import random
from config import *


class Player:  # Object representing a game participant
    def __init__(self, index, strategy, AccPayOffs=0):
        self.index = index  # Player's index
        self.strategy = strategy  # Chosen strategy: True for cooperation, False for defection
        self.newStrategy = strategy  # Updated strategy (after evolution)
        self.AccPayOffs = AccPayOffs  # Accumulated payoff
        self.IsActivate = False  # Whether the player has been activated

    def __str__(self):
        return "Index: {}; Strategy: {}; AccPayOffs: {}".format(self.index, self.strategy, self.AccPayOffs)


def playersInit():
    players.clear()
    half_N = N // 2
    indices = list(range(N))
    random.shuffle(indices)
    for i in range(half_N):
        players.append(Player(indices[i], True, 0))  # Cooperative players
    for i in range(half_N, N):
        players.append(Player(indices[i], False, 0))  # Defective players
    return players
