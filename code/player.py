import random
from config import *


class Player:  # 博弈双方对象
    def __init__(self, index, strategy, AccPayOffs=0):
        self.index = index  # 序号
        self.strategy = strategy  # 选择的博弈策略，True表示合作，False表示对抗
        self.newStrategy = strategy  # 策略更新
        self.AccPayOffs = AccPayOffs  # 累计报酬
        self.IsActivate = False  # 这个点是否被激活过了

    def __str__(self):
        return "Index: {}; Strategy: {}; AccPayOffs: {}".format(self.index, self.strategy, self.AccPayOffs)


def playersInit():
    players.clear()
    half_N = N // 2
    indices = list(range(N))
    random.shuffle(indices)
    for i in range(half_N):
        players.append(Player(indices[i], True, 0))  # 合作玩家
    for i in range(half_N, N):
        players.append(Player(indices[i], False, 0))  # 背叛玩家
    return players
