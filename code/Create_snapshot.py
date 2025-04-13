import copy
import random
import networkx as nx
import networkt as nt
import numpy as np
import matplotlib.pyplot as plt
from config import *
from math import ceil
from QueueAndStack import Queue, Stack


def to_full_numpy_matrix(_NOCs):
    mat = np.zeros((N, N))
    for i in _NOCs.edges:
        mat[i[0], i[1]] = 1
        mat[i[1], i[0]] = 1
    return mat


def Largest_CC(G):
    largest_cc = max(nx.connected_components(G), key=len)  # 获取最大连通子图
    return G.subgraph(list(largest_cc))


def draw_graph(NOC1, NOC2, _nocName, _f):
    """
    双图展示绘制功能
    """
    if flag_draw == 1:
        fig, axs = plt.subplots(1, 2, figsize=(16, 8))
        # 在第一个子图中展示第一个网络
        nx.draw(NOC1, ax=axs[0], pos=nx.kamada_kawai_layout(NOC1))
        axs[0].set_title('Static Network')

        # 在第二个子图中展示第二个网络
        nx.draw(NOC2, ax=axs[1], pos=nx.kamada_kawai_layout(NOC1))
        axs[1].set_title(_nocName+" "+str(_f))

        plt.show()
    elif flag_draw == 2:
        nx.write_gexf(nx.from_numpy_array(to_full_numpy_matrix(NOC2)), "./draw_gexf/"+_nocName+'_'+str(_f)+".gexf")
        # nx.write_gpickle(nx.from_numpy_array(to_full_numpy_matrix(NOC2)),"./draw_gpickle/" + _nocName + '_' + str(_f) + ".gpickle")
    else:
        pass


def print_graph_properties(graphtype, frac, GName, G):
    """
    输出给定图G的属性
    顶点数(Number of vertices)：图中节点的数量
    边数(Number of edges)：图中连接两个节点的边的数量
    度(Degree)：一个节点的度是指与该节点相连的边的数量
    联通性(Connectivity)：图是否连通，即图中任意两点是否存在一条路径连接它们
    密度(Density)：图中边的数量与最大可能边数的比值
    直径(Diameter)：图中最长路径的长度
    平均距离(Average distance)：图中所有节点对间的平均距离
    中心性(Centrality)：节点在图中的重要程度，常用的中心性度量包括度中心性、接近中心性和介数中心性
    团(Clique)：图中一个完全子图
    模块度(Modularity)：图中社团结构的程度
    """
    # print(G.edges())
    # for i in range(N):
    #     print(G.degree[i])

    # 输出图的属性
    if flag_complex:
        print()
        print("|------------------------------------------------------------------|")
        print()
        print("     {} {}({}): {}".format(graphtype, GName, frac, G))
        print("     Density:", nx.density(G))
        if nx.is_connected(G):
            print("     Graph is connected")
            print("     Diameter:", nx.diameter(G))
            print("     Average distance:", nx.average_shortest_path_length(G))
        else:
            print("     Graph is not connected, number_connected_components:", nx.number_connected_components(G))
            print("     Diameter: Not defined for not connected graph")
            print("     Average distance: Not defined for not connected graph")
        # degree_centrality = nx.degree_centrality(G)
        # print("Degree centrality of each vertex:")
        # for vertex, value in degree_centrality.items():
        #     print("Vertex:", vertex, "Degree centrality:", value)
        print("     Clustering coefficient:", nx.average_clustering(G))
        # print("Modularity:", nx.algorithms.community.modularity(G))
    else:
        print("     {} {}({}): {}".format(graphtype, GName, frac, G))


def getName(graphtype, f, subNet):
    params_dict = {
        'SF': str(m) + "m",
        'ER': str(ER_p) + "p",
        'CSMSF': str(m) + "m_" + str(gamma) + "gamma",
        'RR': str(d) + "d"
    }

    name = f"{graphtype}_{N}N_{params_dict[graphtype]}_{f}f_{subNet}"
    return name


def delete01degree(G):
    new_G = G.copy()
    while True:
        to_remove = [node for node in new_G.nodes() if new_G.degree[node] <= 1]
        if len(to_remove) == 0:
            break
        new_G.remove_nodes_from(to_remove)
    return new_G


def link_neighbour(graphtype, node, ActivateEdgesNum, subgraphMaxEdgesNum, subgraphEdges):
    neighbors = np.random.choice(list(NOCs[graphtype].neighbors(node)), replace=False,
                                 size=NOCs[graphtype].degree(node))  # 在搜索邻居的时候需要随机打乱邻居
    for id1 in range(NOCs[graphtype].degree(node)):
        if ActivateEdgesNum >= subgraphMaxEdgesNum:
            break
        for id2 in range(NOCs[graphtype].degree(node)):
            if ActivateEdgesNum >= subgraphMaxEdgesNum:
                break
            if neighbors[id1] != neighbors[id2] \
                    and NOCs[graphtype].has_edge(neighbors[id1], neighbors[id2]) \
                    and (neighbors[id1], neighbors[id2]) not in subgraphEdges \
                    and (neighbors[id2], neighbors[id1]) not in subgraphEdges:
                subgraphEdges.append((neighbors[id1], neighbors[id2]))
                ActivateEdgesNum += 1
    return ActivateEdgesNum


# -----------创建不同的时序网络-------------#
def random_reference(graphtype, subgraphMaxEdgesNum):
    edges = list(NOCs[graphtype].edges())
    subgraphIndices = np.random.choice(len(edges), size=ceil(subgraphMaxEdgesNum), replace=False)
    subgraphEdges = [edges[i] for i in subgraphIndices]
    return subgraphEdges


def multi_structure(graphtype, subgraphMaxEdgesNum, cluster):
    ActivateEdgesNum = 0  # 激活边的数量
    subgraphEdges = []  # 子图的边
    # while ActivateEdgesNum < subgraphMaxEdgesNum:  # 这个while没用
    nodes = np.random.choice(list(NOCs[graphtype].nodes()), size=NOCs[graphtype].number_of_nodes(), replace=False)
    for node in nodes:  # nodes相当于进行了多次不放回抽样
        if ActivateEdgesNum >= subgraphMaxEdgesNum:
            break
        neighbors = np.random.choice(list(NOCs[graphtype].neighbors(node)), replace=False, size=NOCs[graphtype].degree(node))  # 在搜索邻居的时候需要随机打乱邻居
        for id in range(NOCs[graphtype].degree(node)):  # 自身的邻居
            if ActivateEdgesNum >= subgraphMaxEdgesNum:
                break
            if (node, neighbors[id]) not in subgraphEdges and (neighbors[id], node) not in subgraphEdges:
                subgraphEdges.append((node, neighbors[id]))
                ActivateEdgesNum += 1
        if cluster:  # 如果要构成cluster，则将邻居之间的点也都连接上
            ActivateEdgesNum = link_neighbour(graphtype, node, ActivateEdgesNum, subgraphMaxEdgesNum, subgraphEdges)
    return subgraphEdges


def single_structure(graphtype, subgraphMaxEdgesNum, cluster):
    ActivateEdgesNum = 0  # 激活边的数量
    subgraphEdges = []  # 子图的边
    nodeQueue = Queue()  # 用于当前选择的node队列
    # 随机抽一个node，从这个node开始向四周扩散，如果不够，则重新随机抽取node
    nodes = np.random.choice(list(NOCs[graphtype].nodes()), size=NOCs[graphtype].number_of_nodes(), replace=False)
    for rootNode in nodes:
        isActivate = np.zeros(N)  # 记录已经激活的节点，保证严格non-cluster模式
        nodeQueue.push_back(rootNode)
        isActivate[rootNode] = 1
        while ActivateEdgesNum < subgraphMaxEdgesNum:
            if nodeQueue.empty():
                break
            node = nodeQueue.front()
            nodeQueue.pop()
            neighbors = np.random.choice(list(NOCs[graphtype].neighbors(node)), replace=False, size=NOCs[graphtype].degree(node))  # 在搜索邻居的时候需要随机打乱邻居
            for id in range(NOCs[graphtype].degree(node)):  # 自身的邻居
                if (not cluster) and (isActivate[neighbors[id]]):  # non-cluster模式下，激活过的节点不要了
                    continue
                if ActivateEdgesNum >= subgraphMaxEdgesNum:
                    break
                if (node, neighbors[id]) not in subgraphEdges and (neighbors[id], node) not in subgraphEdges:
                    subgraphEdges.append((node, neighbors[id]))
                    nodeQueue.push_back(neighbors[id])
                    isActivate[neighbors[id]] = 1
                    ActivateEdgesNum += 1
            if cluster:  # 如果要构成cluster，则将邻居之间的点也都连接上
                ActivateEdgesNum = link_neighbour(graphtype, node, ActivateEdgesNum, subgraphMaxEdgesNum, subgraphEdges)
    return subgraphEdges


def delete_diff_edge(graphtype, deleteType, maxDelete=1000):
    DeleteEdgesNum = 0  # 删除边的数量
    G = NOCs[graphtype].copy()
    hub = 0
    for node in G.nodes():
        hub = max(hub, G.degree(node))
    tau = 1.3
    phi = 7
    non_continue = True  # 是否循环删边
    if non_continue:  # 不循环删边
        edges = list(G.edges())
        random.shuffle(edges)
        delete_edge = []
        for edge in edges:
            if DeleteEdgesNum >= maxDelete:
                break
            d1 = max(G.degree(edge[0]), G.degree(edge[1]))
            d2 = min(G.degree(edge[0]), G.degree(edge[1]))

            if (deleteType == 'E13' and d1 >= hub / phi and d2 * tau <= d1) or \
                    (deleteType == 'E23' and hub / phi <= d1 < d2 * tau) or \
                    (deleteType == 'E14' and hub / phi > d1 >= d2 * tau) or \
                    (deleteType == 'E24' and d1 < hub / phi and d2 * tau > d1) or \
                    (deleteType == 'E1' and d1 >= d2 * tau) or \
                    (deleteType == 'E2' and d1 < d2 * tau) or \
                    (deleteType == 'E3' and d1 >= hub / phi) or \
                    (deleteType == 'E4' and d1 < hub / phi) or \
                    (deleteType == 'Random'):
                delete_edge.append((edge[0], edge[1]))
        # print(len(delete_edge))
        for edge in delete_edge:
            if DeleteEdgesNum >= maxDelete:
                break
            G.remove_edge(edge[0], edge[1])
            DeleteEdgesNum += 1
            if not nx.is_connected(G):  # 保证网络连通性
                G.add_edge(edge[0], edge[1])
                DeleteEdgesNum -= 1

    else:
        while DeleteEdgesNum < maxDelete:
            flag = 0
            edges = list(G.edges())
            random.shuffle(edges)
            for edge in edges:
                if DeleteEdgesNum >= maxDelete:
                    break
                d1 = max(G.degree(edge[0]), G.degree(edge[1]))
                d2 = min(G.degree(edge[0]), G.degree(edge[1]))

                if (deleteType == 'E13' and d1 >= hub / phi and d2 * tau <= d1) or \
                        (deleteType == 'E23' and hub / phi <= d1 < d2 * tau) or \
                        (deleteType == 'E14' and hub / phi > d1 >= d2 * tau) or \
                        (deleteType == 'E24' and d1 < hub / phi and d2 * tau > d1) or \
                        (deleteType == 'E1' and d1 >= d2 * tau) or \
                        (deleteType == 'E2' and d1 < d2 * tau) or \
                        (deleteType == 'E3' and d1 >= hub / phi) or \
                        (deleteType == 'E4' and d1 < hub / phi) or \
                        (deleteType == 'Random'):
                        flag += 1
                        G.remove_edge(edge[0], edge[1])
                        DeleteEdgesNum += 1
                        if not nx.is_connected(G):  # 保证网络连通性
                            flag -= 1
                            G.add_edge(edge[0], edge[1])
                            DeleteEdgesNum -= 1
            if flag == 0:  # 没有删除的边
                break
    return G.edges()


def create_structural_temporal_network(graphtype, f, snapshotName):
    graphList = []
    subgraphMaxEdgesNum = f * NOCs[graphtype].number_of_edges()

    for t in range(snapshotNum):
        delete_edge_number = 118
        structure_function_map = {
            "Random": (random_reference, (graphtype, subgraphMaxEdgesNum)),
            "Single-star": (single_structure, (graphtype, subgraphMaxEdgesNum, False)),
            "Single-cluster": (single_structure, (graphtype, subgraphMaxEdgesNum, True)),
            "Multi-star": (multi_structure, (graphtype, subgraphMaxEdgesNum, False)),
            "Multi-cluster": (multi_structure, (graphtype, subgraphMaxEdgesNum, True)),
            "Delete-E1": (delete_diff_edge, (graphtype, 'E1', delete_edge_number)),
            "Delete-E2": (delete_diff_edge, (graphtype, 'E2', delete_edge_number)),
            "Delete-E3": (delete_diff_edge, (graphtype, 'E3', delete_edge_number)),
            "Delete-E4": (delete_diff_edge, (graphtype, 'E4', delete_edge_number)),
            "Delete-E13": (delete_diff_edge, (graphtype, 'E13', delete_edge_number)),
            "Delete-E23": (delete_diff_edge, (graphtype, 'E23', delete_edge_number)),
            "Delete-E14": (delete_diff_edge, (graphtype, 'E14', delete_edge_number)),
            "Delete-E24": (delete_diff_edge, (graphtype, 'E24', delete_edge_number)),
            "Delete-Random": (delete_diff_edge, (graphtype, 'Random', delete_edge_number)),
        }

        if snapshotName in structure_function_map:
            function, args = structure_function_map[snapshotName]
            subgraphEdges = function(*args)
        else:
            raise ValueError("Invalid snapshotName")
        Wt = nx.edge_subgraph(NOCs[graphtype], subgraphEdges)
        if flag_change_snapshot:
            change_function_map = {
                0: (delete01degree, Wt)
            }
            function, args = change_function_map[0]
            Wt = function(args)
        draw_graph(NOCs[graphtype], Wt, snapshotName, f)
        print_graph_properties(graphtype, f, snapshotName, Wt)
        graphList.append(to_full_numpy_matrix(Wt))

        if snapshotName.startswith('Delete'):  # 不具有随机性的网络只需要生成一次
            break

    if flag_save:
        filename = getName(graphtype, f, snapshotName)
        if flag_change_snapshot:
            np.save("./" + snapshot_name + "/" + filename + "_change.npy", np.array(graphList))
        else:
            np.save("./" + snapshot_name + "/" + filename + ".npy", np.array(graphList))


def save_graph(graph, graph_type, params):
    tempMat = [to_full_numpy_matrix(graph)]
    param_str = "_".join([f"{p}{s}" for p, s in params])
    np.save(f"./{snapshot_name}/{getName(graph_type, 0, 'Static')}.npy", np.array(tempMat))


def load_graph(graph_type, params):
    param_str = "_".join([f"{p}{s}" for p, s in params])
    return nx.from_numpy_array(np.load(f"./{snapshot_name}/{getName(graph_type, 0, 'Static')}.npy", allow_pickle=True)[0])


def get_graphs(graph_types):
    NOCs = {}
    graph_params = {
        'SF': (nx.barabasi_albert_graph, [(N, 'N'), (m, 'm')]),
        'ER': (nx.erdos_renyi_graph, [(N, 'N'), (ER_p, 'p')]),
        'CSMSF': (nt.connect_static_model_scale_free_graph, [(N, 'N'), (m, 'm'), (gamma, 'gamma')]),
        'RR': (nx.random_regular_graph, [(d, 'd'), (N, 'N')])
    }

    for graph_type in graph_types:
        func, params = graph_params[graph_type]
        if flag_static:
            graph = func(*map(lambda x: x[0], params))
            if flag_save:
                save_graph(graph, graph_type, params)
        else:
            graph = load_graph(graph_type, params)
        NOCs[graph_type] = graph

    return NOCs


if __name__ == '__main__':
    NOCs = get_graphs(graphType)

    for i in graphType:
        print_graph_properties(i, 1, i, NOCs[i])
        if flag_draw == 2:
            nx.write_gexf(NOCs[i], "./draw_gexf/static_{}.gexf".format(i))

    create_snapshot = True
    if create_snapshot:
        for f in flist:
            for gt in graphType:
                for sn in subNet:
                    create_structural_temporal_network(gt, f, sn)

