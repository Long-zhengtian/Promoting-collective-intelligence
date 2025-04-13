import numpy as np

dataPath = "/Users/tianzhenglong/Documents/GitHub/Structural-temporal-network/code_final/data/"
snapshotPath = "/Users/tianzhenglong/资料/科研/课题：结构时序网络/final_result"
players = []
# seed_value = 2022
snapshot_name = 'snapshot'
result_name = 'result'

# 网络参数
N = 400  # 网络规模
graphType = ['SF']
# subNet = ['Static', 'Random', 'Single-cluster', 'Multi-cluster', 'Single-star', 'Multi-star']
subNet = ['Random', 'Single-cluster', 'Multi-cluster', 'Single-star', 'Multi-star']


snapshotNum = 200
m = 4  # BA模型参数
gamma = 2.3  # static model参数
ER_p = 0.01  # ER图参数
d = 8  # RR图参数
s = 2  # 费米狄拉克函数选择强度，强选择

# 演化参数
G1 = 50000  # 前置演化
G2 = 2000  # 平均演化

DiffGraph = 1  # 跑DiffGraph个不同的图，取平均值
EG_Rounds = 100  # 每个图跑ER_Rounds次
b_n = 16
blist = np.linspace(1.0, 2.5, b_n)

glist = [100, 500]
flist = [0.3]

# 控制参数
flag_save = True  # 当前的snapshot是否保存
flag_draw = 3  # 1 matplotlib绘制，2 gephi绘制, 3 不绘制
flag_complex = False  # 每个snapshot输出信息量：复杂or精简
flag_static = True  # 是否重新创建static网络
flag_change_snapshot = False  # 是否对生成的时序网络进行改动


# 博弈参数
class PayOff_PD:  # 囚徒博弈的得失情况
    def __init__(self, b):
        self.T = b
        self.R = 1
        self.P = 0
        self.S = 0


class PayOff_SG:  # 雪堆博弈的得失情况
    def __init__(self, r=0.5):
        beta = (1 / r + 1)/2
        self.T = beta
        self.R = beta - 0.5
        self.P = 0
        self.S = beta - 1
