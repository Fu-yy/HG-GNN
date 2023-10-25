import pickle
import math
from operator import itemgetter
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import numpy as np
import pandas as pd 
#from pandarallel import pandarallel
import pickle
from tqdm import tqdm 
import dgl


def userTopItems(dataset, K=10):
    '''
    功能是根据给定的数据集，计算每个用户的Top K推荐物品，并将结果保存到文件中。

    函数 userTopItems 的输入参数为 dataset（数据集名称）和可选参数 K（默认为10，表示每个用户的Top K推荐物品数量）。

    从文件中加载训练数据集 train.pkl，该文件包含用户的历史会话数据。
    创建字典 u_dict 来存储用户的历史会话数据和字典 item_pop 来存储物品的流行度。
    统计物品的流行度：遍历每个用户的历史会话数据，对于每个会话中的物品，将其流行度加一。
    根据物品的流行度计算用户对物品的喜好程度：遍历每个用户的历史会话数据，对于每个会话中的物品，根据物品的流行度计算用户对该物品的喜好程度，并将结果存储到字典 u_dict 中。
    创建字典 user_topK 来存储每个用户的Top K推荐物品。
    对于每个用户，根据其对物品的喜好程度，选取Top K热门物品和Top K冷门物品作为推荐物品，并将结果存储到字典 user_topK 中。
    将用户的Top K推荐物品保存到文件 userTopItems.pkl 中。
    请注意，代码中使用了 tqdm 来显示进度条，以便在处理大量数据时能够实时显示进度。另外，代码中有两处文件路径需要根据实际情况进行修改，分别是训练数据集文件路径和保存用户Top K物品的文件路径。
    Args:
        dataset:
        K:

    Returns:

    '''


    # 从文件中加载训练数据
    with open(f'E:/MyCode/PycharmCode/HG-GNN/data_processor/dataset/{dataset}/train.pkl', 'rb') as f:
        session_data = pickle.load(f)

    # 创建字典来存储用户的历史会话数据和物品的流行度
    u_dict = dict()  # 存储用户的历史会话数据
    item_pop = dict()  # 存储物品的流行度

    # 统计物品的流行度
    for uid in tqdm(session_data):
        u_sess = session_data[uid]
        for sess in u_sess:
            for vid in sess:
                item_pop.setdefault(vid, 0)
                item_pop[vid] += 1

    # 根据物品的流行度计算用户对物品的喜好程度
    for uid in tqdm(session_data):
        u_sess = session_data[uid]
        u_dict.setdefault(uid, dict())
        for sess in u_sess:
            for vid in sess:
                u_dict[uid].setdefault(vid, 0)
                u_dict[uid][vid] += (item_pop[vid] * 0.75)

    user_topK = {}  # 存储每个用户的Top K物品

    # 根据用户对物品的喜好程度，选取Top K热门物品和Top K冷门物品作为用户的推荐物品
    for user in u_dict:
        hot_items = [key for key, value in sorted(u_dict[user].items(), key=itemgetter(1), reverse=True)[:K]]
        cold_items = [key for key, value in sorted(u_dict[user].items(), key=itemgetter(1), reverse=False)[:K]]
        user_topK[user] = list(set(hot_items).union(set(cold_items)))

    # 将用户的Top K物品保存到文件中
    with open(f'E:/MyCode/PycharmCode/HG-GNN/data_processor/dataset/{dataset}/userTopItems.pkl', 'wb') as f:
        pickle.dump(user_topK, f)


def itemTopUsers(dataset, K=10):

    '''

    功能是根据给定的数据集，计算每个物品的Top K推荐用户，并将结果保存到文件中。

    函数 itemTopUsers 的输入参数为 dataset（数据集名称）和可选参数 K（默认为10，表示每个物品的Top K推荐用户数量）。

    从文件中加载训练数据集 train.pkl，该文件包含物品的历史会话数据。
    创建字典 v_dict 来存储物品的历史会话数据和字典 user_active 来存储用户的活跃度。
    统计用户的活跃度：遍历每个用户的历史会话数据，计算每个用户的活跃度（会话数量的总和），并将结果存储到字典 user_active 中。
    根据用户的活跃度计算物品的受欢迎程度：遍历每个用户的历史会话数据，对于每个会话中的物品，根据用户的活跃度计算物品的受欢迎程度，并将结果存储到字典 v_dict 中。
    创建字典 item_topK 来存储每个物品的Top K推荐用户。
    对于每个物品，根据其受欢迎程度，选取Top K活跃用户和Top K不活跃用户作为推荐用户，并将结果存储到字典 item_topK 中。
    将物品的Top K推荐用户保存到文件 itemTopUtems.pkl 中。
    请注意，代码中使用了 tqdm 来显示进度条，以便在处理大量数据时能够实时显示进度。另外，代码中有两处文件路径需要根据实际情况进行修改，分别是训练数据集文件路径和保存物品Top K用户的文件路径。
    Args:
        dataset:
        K:

    Returns:

    '''
    # 从文件中加载训练数据
    with open(f'E:/MyCode/PycharmCode/HG-GNN/data_processor/dataset/{dataset}/train.pkl', 'rb') as f:
        session_data = pickle.load(f)

    # 创建字典来存储物品的历史会话数据和用户的活跃度
    v_dict = dict()  # 存储物品的历史会话数据
    user_active = dict()  # 存储用户的活跃度

    # 统计用户的活跃度
    for uid in tqdm(session_data):
        u_sess = session_data[uid]
        user_active[uid] = sum([len(sess) for sess in u_sess])

    # 根据用户的活跃度计算物品的受欢迎程度
    for uid in tqdm(session_data):
        u_sess = session_data[uid]

        for sess in u_sess:
            for vid in sess:
                v_dict.setdefault(vid, dict())
                v_dict[vid].setdefault(uid, 0)
                v_dict[vid][uid] += (user_active[uid] * 0.75)

    item_topK = {}  # 存储每个物品的Top K用户

    # 根据物品的受欢迎程度，选取Top K活跃用户和Top K不活跃用户作为物品的推荐用户
    for item in v_dict:
        hot_users = [key for key, value in sorted(v_dict[item].items(), key=itemgetter(1), reverse=True)[:K]]
        cold_users = [key for key, value in sorted(v_dict[item].items(), key=itemgetter(1), reverse=False)[:K]]
        item_topK[item] = list(set(hot_users).union(set(cold_users)))

    # 将物品的Top K用户保存到文件中
    with open(f'E:/MyCode/PycharmCode/HG-GNN/data_processor/dataset/{dataset}/itemTopUtems.pkl', 'wb') as f:
        pickle.dump(item_topK, f)


def userCF(dataset):
    """
    计算用户之间的相似度
        功能是计算用户之间的相似度，并将每个用户的Top K相似用户保存到文件中。

        函数 userCF 的输入参数为 dataset（数据集名称）。

        创建字典 vid_user，用于存储物品和对应的用户集合。
        创建字典 user_sim_matrix，用于存储用户之间的相似度矩阵。
        创建字典 uid_vcount，用于存储用户和其观看过的物品数量。
        从文件中加载训练数据集 train.pkl。
        构建物品和对应的用户集合的字典 vid_user，以及用户和其观看过的物品数量的字典 uid_vcount。
        计算用户之间的相似度：遍历物品和对应的用户集合，对于每对不同的用户，增加其相似度值。
        根据观看过的物品数量归一化相似度，得到最终的用户相似度矩阵。
        创建字典 user_topK，用于存储每个用户的Top K相似用户。
        选取每个用户的Top K相似用户，并将结果存储到字典 user_topK 中。
        将用户的Top K相似用户保存到文件 u2u_sim.pkl 中。
        请注意，代码中使用了 tqdm 来显示进度条，以便在处理大量数据时能够实时显示进度。另外，代码中有一处文件路径需要根据实际情况进行修改，即保存用户相似度矩阵的文件路径。
    """
    vid_user = {}  # 存储物品和对应的用户集合
    user_sim_matrix = {}  # 存储用户之间的相似度矩阵
    uid_vcount = {}  # 存储用户和其观看过的物品数量
    with open(f'E:/MyCode/PycharmCode/HG-GNN/data_processor/dataset/{dataset}/train.pkl', 'rb') as f:
        session_data = pickle.load(f)  # 加载训练数据集

    # 构建物品和对应的用户集合的字典，以及用户和其观看过的物品数量的字典
    for uid in tqdm(session_data):
        u_sess = session_data[uid]
        uid_vcount.setdefault(uid, set())

        for sess in u_sess:
            for vid in sess:
                if vid not in vid_user:
                    vid_user[vid] = set()
                vid_user[vid].add(uid)
                uid_vcount[uid].add(vid)

    # 计算用户之间的相似度
    for vid, users in tqdm(vid_user.items()):
        for u in users:
            for v in users:
                if u == v:
                    continue

                user_sim_matrix.setdefault(u, {})
                user_sim_matrix[u].setdefault(v, 0)
                user_sim_matrix[u][v] += (1 / len(users))

    # 根据观看过的物品数量归一化相似度得到最终的用户相似度矩阵
    for u, related_users in user_sim_matrix.items():
        for v, count in related_users.items():
            user_sim_matrix[u][v] = count / math.sqrt(len(uid_vcount[u]) * len(uid_vcount[v]))

    user_topK = {}  # 存储每个用户的Top K相似用户

    # 选取每个用户的Top K相似用户
    for user in user_sim_matrix:
        user_topK[user] = sorted(user_sim_matrix[user].items(), key=itemgetter(1), reverse=True)[:100]

    # 将用户的Top K相似用户保存到文件中
    with open(f'E:/MyCode/PycharmCode/HG-GNN/data_processor/dataset/{dataset}/u2u_sim.pkl', 'wb') as f:
        pickle.dump(user_topK, f)


def itemCF(dataset):
    """
    计算物品之间的相似度
    `itemCF` 函数用于计算物品之间的相似度，并将每个物品的Top K相似物品保存到文件中。

    变量和方法功能的注释如下：

    - `uid_item`：字典，存储每个用户和其观看的物品集合。
    - `item_sim_matrix`：字典，存储物品之间的相似度矩阵。
    - `vid_ucount`：字典，存储每个物品对应的用户数量。

    加载训练数据集：
    - 从文件中加载训练数据集 `train.pkl`。

    构建用户和物品的字典：
    - 遍历每个用户的观看历史，在 `uid_item` 字典中存储每个用户和其观看的物品集合。
    - 在 `vid_ucount` 字典中存储每个物品对应的用户数量。

    计算物品之间的相似度：
    - 遍历每个用户的观看历史，对于每对不同的物品，增加其相似度值。

    归一化相似度得到最终的物品相似度矩阵：
    - 遍历物品相似度矩阵 `item_sim_matrix`，计算每对物品之间的相似度，并进行归一化。

    选取每个物品的Top K相似物品：
    - 遍历物品相似度矩阵 `item_sim_matrix`，对于每个物品，选取相似度最高的前 K 个物品。

    保存结果到文件：
    - 将每个物品的Top K相似物品保存到文件 `i2i_sim
    """
    uid_item = {}  # 存储用户和其观看的物品集合
    item_sim_matrix = {}  # 存储物品之间的相似度矩阵
    vid_ucount = {}  # 存储物品和对应的用户数量
    with open(f'E:/MyCode/PycharmCode/HG-GNN/data_processor/dataset/{dataset}/train.pkl', 'rb') as f:
        session_data = pickle.load(f)  # 加载训练数据集

    # 构建用户和其观看的物品集合的字典，以及物品和对应的用户数量的字典
    for uid in tqdm(session_data):
        u_sess = session_data[uid]
        uid_item[uid] = set()

        for sess in u_sess:
            for vid in sess:
                uid_item[uid].add(vid)
                vid_ucount.setdefault(vid, set())
                vid_ucount[vid].add(uid)

    # 计算物品之间的相似度
    for uid, items in tqdm(uid_item.items()):
        for v in items:
            for _v in items:
                if _v == v:
                    continue

                item_sim_matrix.setdefault(v, {})
                item_sim_matrix[v].setdefault(_v, 0)
                item_sim_matrix[v][_v] += (1 / len(items))

    # 根据用户数量归一化相似度得到最终的物品相似度矩阵
    for v, related_items in item_sim_matrix.items():
        for _v, count in related_items.items():
            item_sim_matrix[v][_v] = count / math.sqrt(len(vid_ucount[v]) * len(vid_ucount[_v]))

    item_topK = {}  # 存储每个物品的Top K相似物品

    # 选取每个物品的Top K相似物品
    for item in item_sim_matrix:
        item_topK[item] = sorted(item_sim_matrix[item].items(), key=itemgetter(1), reverse=True)[:200]

    # 将物品的Top K相似物品保存到文件中
    with open(f'E:/MyCode/PycharmCode/HG-GNN/data_processor/dataset/{dataset}/i2i_sim.pkl', 'wb') as f:
        pickle.dump(item_topK, f)


def uui_graph(dataset_name, sample_size, topK, add_u=True, add_v=True):
    """
    根据给定的数据集名称、样本大小和topK值构建图结构。

    参数:
        dataset_name (str): 数据集名称
        sample_size (int): 样本大小
        topK (int): topK值
        add_u (bool): 是否添加用户节点，默认为True
        add_v (bool): 是否添加物品节点，默认为True

    返回:
        G (DGLGraph): 构建的图结构
        item_num (int): 物品节点数量

    功能是根据给定的数据集名称、样本大小和topK值构建图结构。具体的操作如下：

    导入所需的库。
    定义变量pre、nxt、src_v和dst_u，它们分别表示前驱节点、后继节点、物品节点和用户节点的列表。
    调用itemCF函数构建物品之间的相似性关系。
    调用userCF函数构建用户之间的相似性关系。
    使用pickle库从文件中加载图数据和邻接矩阵数据。
    对图进行采样，将采样后的节点和边添加到pre、nxt、src_v和dst_u列表中。
    使用pickle库从文件中加载用户之间的相似性信息和物品之间的相似性信息。
    根据物品相似性信息生成物品节点之间的关系，将结果保存到topv_src和topv_dst列表中。
    根据用户相似性信息生成用户节点之间的关系，将结果保存到u_src和u_dst列表中。
    计算邻接矩阵的平均密度并输出。
    计算物品节点的数量。
    根据需要是否添加用户节点和物品节点的边关系。
    创建空的DGL图。
    将之前收集的节点和边添加到DGL图中。
    添加自环边。
    返回构建的图结构和物品节点的数量。
    """

    pre = []  # 前驱节点列表
    nxt = []  # 后继节点列表
    src_v = []  # 物品节点列表
    dst_u = []  # 用户节点列表

    # 构建物品之间的相似性关系
    itemCF(dataset_name)

    # 构建用户之间的相似性关系
    userCF(dataset_name)

    # 从pickle文件中加载图数据
    with open(f'E:/MyCode/PycharmCode/HG-GNN/data_processor/dataset/{dataset_name}/train.pkl', 'rb') as f:
        graph = pickle.load(f)

    # 从pickle文件中加载邻接矩阵数据
    with open(f'E:/MyCode/PycharmCode/HG-GNN/data_processor/dataset/{dataset_name}/adj_{sample_size}.pkl', 'rb') as f:
        adj = pickle.load(f)
    adj_in = adj[0]
    adj_out = adj[1]
    print('adj_in:', len(adj_in))
    print('adj_out:', len(adj_out))

    ## 对图进行采样
    for i in range(len(adj_in)):
        if i == 0:
            continue
        _pre = []
        _nxt = []
        for item in adj_in[i]:
            _pre.append(i)
            _nxt.append(item)
        pre += _pre
        nxt += _nxt
    o_pre = []
    o_nxt = []
    for i in range(len(adj_out)):
        if i == 0:
            continue
        _pre = []
        _nxt = []
        for item in adj_out[i]:
            _pre.append(i)
            _nxt.append(item)
        o_pre += _pre
        o_nxt += _nxt

    # 构建用户-物品之间的关系
    for u in tqdm(graph, desc='build the graph...', leave=False):
        u_seqs = graph[u]
        for s in u_seqs:
            pre += s[:-1]
            nxt += s[1:]
            dst_u += [u for _ in s]
            src_v += s

    # 从pickle文件中加载用户相似性信息
    with open(f'E:/MyCode/PycharmCode/HG-GNN/data_processor/dataset/{dataset_name}/u2u_sim.pkl', 'rb') as f:
        u2u_sim = pickle.load(f)

    # 从pickle文件中加载物品相似性信息
    with open(f'E:/MyCode/PycharmCode/HG-GNN/data_processor/dataset/{dataset_name}/i2i_sim.pkl', 'rb') as f:
        i2i_sim = pickle.load(f)

    topv_src = []
    topv_dst = []
    count_v = 0
    for v in tqdm(i2i_sim, desc='gen_seq...', leave=False):
        tmp_src = []
        tmp_dst = []

        exclusion = adj_in[v] + adj_out[v]
        for (vid, value) in i2i_sim[v][:topK][:int(len(exclusion))]:
            if vid not in exclusion:
                tmp_src.append(vid)
                tmp_dst.append(v)
        topv_src += tmp_src
        topv_dst += tmp_dst

    u_src = []
    u_dst = []
    for u in tqdm(u2u_sim, desc='gen_seq...', leave=False):
        tmp_src = []
        tmp_dst = []
        for (uid, value) in u2u_sim[u][:topK]:
            tmp_src.append(uid)
            tmp_dst.append(u)
        u_src += tmp_src
        u_dst += tmp_dst

    count = 0
    for i in adj_in:
        count += len(i)
    print('local ajdency-in:', count / len(adj_in))
    count = 0
    for i in adj_out:
        count += len(i)
    print('local ajdency-out:', count / len(adj_out))

    item_num = max(max(pre), max(nxt)) + 1
    print('addiotn item num', item_num)
    user_num = max(max(u_src), max(u_dst))
    u_src = [u + item_num for u in u_src]
    u_dst = [u + item_num for u in u_dst]
    dst_u = [u + item_num for u in dst_u]

    # 创建空的DGL图
    G = dgl.graph((pre, nxt))
    G = dgl.add_edges(G, nxt, pre)
    G = dgl.add_edges(G, dst_u, src_v)
    G = dgl.add_edges(G, src_v, dst_u)

    if add_u:
        G = dgl.add_edges(G, u_src, u_dst)
        G = dgl.add_edges(G, u_dst, u_src)

    if add_v:
        G = dgl.add_edges(G, topv_src, topv_dst)
        G = dgl.add_edges(G, topv_dst, topv_src)

    G = dgl.add_self_loop(G)

    return G, item_num


def sample_relations(dataset_name, num, sample_size=20):
    """
    根据给定的数据集名称、数量和采样大小构建关系数据。

    参数:
        dataset_name (str): 数据集名称
        num (int): 数据集中的节点数量
        sample_size (int): 采样大小，默认为20

    返回:
        None


        功能是根据给定的数据集名称、数量和采样大小构建关系数据。具体的操作如下：

        创建空的邻接矩阵和关系列表。
        使用pickle库从文件中加载图数据。
        构建关系数据，遍历图数据中的用户序列，提取相邻节点之间的关系，并添加到关系列表中。
        构建邻接矩阵，遍历关系列表，统计每个节点的出边和入边，并保存到邻接矩阵中。
        对邻接矩阵进行排序，获取每个节点的前sample_size个出边和入边。
        对邻接矩阵进行采样，保留每个节点的前sample_size个出边和入边。
        将采样后的邻接矩阵保存到pickle文件。
    """

    # 创建空的邻接矩阵和关系列表
    adj1 = [dict() for _ in range(num)]
    adj2 = [dict() for _ in range(num)]
    adj_in = [[] for _ in range(num)]
    adj_out = [[] for _ in range(num)]
    relation_out = []
    relation_in = []

    # 从pickle文件中加载图数据
    with open(f'E:/MyCode/PycharmCode/HG-GNN/data_processor/dataset/{dataset_name}/train.pkl', 'rb') as f:
        graph = pickle.load(f)

    # 构建关系数据
    for u in tqdm(graph, desc='build the graph...', leave=False):
        u_seqs = graph[u]
        for s in u_seqs:
            for i in range(len(s) - 1):
                relation_out.append([s[i], s[i + 1]])
                relation_in.append([s[i + 1], s[i]])

    # 构建邻接矩阵
    for tup in relation_out:
        if tup[1] in adj1[tup[0]].keys():
            adj1[tup[0]][tup[1]] += 1
        else:
            adj1[tup[0]][tup[1]] = 1
    for tup in relation_in:
        if tup[1] in adj2[tup[0]].keys():
            adj2[tup[0]][tup[1]] += 1
        else:
            adj2[tup[0]][tup[1]] = 1

    # 对邻接矩阵进行排序，获取前sample_size个边
    for t in range(1, num):
        x = [v for v in sorted(adj1[t].items(), reverse=True, key=lambda x: x[1])]
        adj_out[t] = [v[0] for v in x]

    for t in range(1, num):
        x = [v for v in sorted(adj2[t].items(), reverse=True, key=lambda x: x[1])]
        adj_in[t] = [v[0] for v in x]

    # 对邻接矩阵进行采样，保留前sample_size个边
    for i in range(1, num):
        adj_in[i] = adj_in[i][:sample_size]
    for i in range(1, num):
        adj_out[i] = adj_out[i][:sample_size]

    # 将采样后的邻接矩阵保存到pickle文件
    with open(f'E:/MyCode/PycharmCode/HG-GNN/data_processor/dataset/{dataset_name}/adj_{sample_size}.pkl', 'wb') as f:
        pickle.dump([adj_in, adj_out], f)