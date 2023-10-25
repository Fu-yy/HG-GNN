

"""
following the torch.Dataset, we prepare the standard dataset input for Models

"""
import os
import pickle

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import configparser


def gen_seq(data_list):
    """
    生成序列数据。

    参数:
        data_list: 数据列表，包含用户和对应的序列。

    返回值:
        uid: 用户ID列表。
        out_seqs: 输出的序列列表，每个序列去掉最后一个元素。
        label: 序列的标签列表，每个标签是序列的最后一个元素。
    """
    out_seqs = []  # 输出的序列列表，每个序列去掉最后一个元素
    label = []  # 序列的标签列表，每个标签是序列的最后一个元素
    uid = []  # 用户ID列表

    for u in tqdm(data_list, desc='gen_seq...', leave=False):
        u_seqs = data_list[u]  # 获取用户对应的序列
        for seq in u_seqs:
            for i in range(1, len(seq)):
                uid.append(int(u))  # 将用户ID添加到列表中
                out_seqs.append(seq[:-i])  # 将去掉最后一个元素的序列添加到列表中
                label.append([seq[-i]])  # 将最后一个元素作为标签添加到列表中

    return (uid, out_seqs, label)


def common_seq(data_list):
    """
    提取常见的序列数据。

    参数:
        data_list: 数据列表，包含用户和对应的序列。

    返回值:
        final_seqs: 最终的序列列表，每个序列包含用户ID、输入序列和标签。
    """
    out_seqs = []  # 输出的序列列表，每个序列去掉最后一个元素
    label = []  # 序列的标签列表，每个标签是序列的最后一个元素
    uid = []  # 用户ID列表

    for u in tqdm(data_list, desc='gen_seq...', leave=False):
        u_seqs = data_list[u]  # 获取用户对应的序列
        for seq in u_seqs:
            for i in range(1, len(seq)):
                uid.append(int(u))  # 将用户ID添加到列表中
                out_seqs.append(seq[:-i])  # 将去掉最后一个元素的序列添加到列表中
                label.append([seq[-i]])  # 将最后一个元素作为标签添加到列表中

    final_seqs = []
    for i in range(len(uid)):
        final_seqs.append([uid[i], out_seqs[i], label[i]])  # 将用户ID、输入序列和标签组合成一个序列

    return final_seqs


def load_data(dataset, data_path):
    """
    加载数据集。

    参数:
        dataset: 数据集名称。
        data_path: 数据路径。

    返回值:
        train_data: 训练数据。
        test_data: 测试数据。
        max_vid: 最大视频ID。
        max_uid: 最大用户ID。
    """
    # 检查是否已存在序列化的数据文件
    data_path = os.path.join(os.path.dirname(__file__),'dataset')
    if not os.path.exists(os.path.join(data_path, dataset,'train_seq.pkl')):
        # 创建临时文件路径以保存数据
        print('try to build ', os.path.join(data_path, dataset,'train_seq.pkl') )
        with open(os.path.join(data_path, dataset,'train.pkl') , 'rb') as f:
            train_data = pickle.load(f)
        max_vid = 0
        max_uid = 0
        for u in train_data:
            if u > max_uid:
                max_uid = u
            for sess in train_data[u]:
                if max_vid < max(sess):
                    max_vid = max(sess)

        try:
            with open(os.path.join(data_path, dataset, 'all_test.pkl') , 'rb') as f:
                test_data = pickle.load(f)
        except:
            with open(os.path.join(data_path, dataset,'test.pkl')  , 'rb') as f:
                test_data = pickle.load(f)

        train_data = common_seq(train_data)  # 提取训练数据的常见序列
        test_data = common_seq(test_data)  # 提取测试数据的常见序列

        with open(os.path.join(data_path, dataset,'test_seq.pkl') , 'wb') as f:
            pickle.dump(test_data, f)  # 将处理后的测试数据序列化保存

        with open(os.path.join(data_path, dataset,'train_seq.pkl') , 'wb') as f:
            pickle.dump(train_data, f)  # 将处理后的训练数据序列化保存

        return train_data, test_data, max_vid, max_uid

    with open(os.path.join(data_path, dataset,'train_seq.pkl')  , 'rb') as f:
        train_data = pickle.load(f)
    max_vid = 0
    max_uid = 0
    for data in train_data:
        if data[0] > max_uid:
            max_uid = data[0]
        if max_vid < max(data[1]):
            max_vid = max(data[1])
        if max_vid < max(data[2]):
            max_vid = max(data[2])

    with open(os.path.join(data_path, dataset, 'test_seq.pkl') , 'rb') as f:
        test_data = pickle.load(f)
    for data in test_data:
        if data[0] > max_uid:
            max_uid = data[0]
        if max_vid < max(data[1]):
            max_vid = max(data[1])
        if max_vid < max(data[2]):
            max_vid = max(data[2])

    return train_data, test_data, max_vid, max_uid


class SessionDataset(Dataset):
    def __init__(self, data, config, max_len=None):
        """
        会话数据集类，用于处理数据集的加载和预处理。

        参数:
            data: 数据列表，包含用户ID、浏览序列、标签等信息的元组。
            config: 配置信息字典。
            max_len: 序列的最大长度。

        属性:
            data: 数据列表，包含用户ID、浏览序列、标签等信息的元组。
            max_seq_len: 序列的最大长度。

        """
        super(SessionDataset, self).__init__()
        self.data = data
        if max_len:
            self.max_seq_len = max_len
        else:
            self.max_seq_len = config['dataset.seq_len']

    def __len__(self):
        """
        返回数据集的长度。

        返回值:
            数据集的长度。
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        根据索引获取数据集中的样本。

        参数:
            index: 样本索引。

        返回值:
            uid: 用户ID。
            browsed_ids: 浏览的视频ID序列。
            mask: 序列的掩码，用于指示有效元素。
            seq_len: 序列的实际长度。
            label: 标签。
            pos_idx: 正样本的索引。

        """
        data = self.data[index]
        uid = np.array([data[0]], dtype=np.int)  # 用户ID
        browsed_ids = np.zeros((self.max_seq_len), dtype=np.int)  # 浏览的视频ID序列
        seq_len = len(data[1][-self.max_seq_len:])  # 序列的实际长度
        mask = np.array([1 for i in range(seq_len)] + [0 for i in range(self.max_seq_len - seq_len)], dtype=np.int)  # 序列的掩码，用于指示有效元素
        pos_idx = np.array([seq_len - i - 1 for i in range(seq_len)] + [0 for i in range(self.max_seq_len - seq_len)], dtype=np.int)  # 正样本的索引
        browsed_ids[:seq_len] = np.array(data[1][-self.max_seq_len:])  # 将浏览的视频ID序列填充到对应位置
        seq_len = np.array(seq_len, dtype=np.int)  # 序列的实际长度
        label = np.array(data[2], dtype=np.int)  # 标签

        return uid, browsed_ids, mask, seq_len, label, pos_idx

class SessionGraphDataset(Dataset):
    def __init__(self, data, config, max_len=None):
        """
        会话图数据集类，用于处理带有图结构的数据集的加载和预处理。

        参数:
            data: 数据列表，包含用户ID、浏览序列、标签等信息的元组。
            config: 配置信息字典。
            max_len: 序列的最大长度。

        属性:
            data: 数据列表，包含用户ID、浏览序列、标签等信息的元组。
            max_seq_len: 序列的最大长度。

        """
        super(SessionGraphDataset, self).__init__()
        self.data = data
        if max_len:
            self.max_seq_len = max_len
        else:
            self.max_seq_len = config['dataset.seq_len']

    def __len__(self):
        """
        返回数据集的长度。

        返回值:
            数据集的长度。
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        根据索引获取数据集中的样本。

        参数:
            index: 样本索引。

        返回值:
            uid: 用户ID。
            global_ids: 全局节点ID序列。
            mask: 序列的掩码，用于指示有效元素。
            seq_len: 序列的实际长度。
            label: 标签。
            local_nodes: 本地节点ID序列，包含重复的节点。
            local_ids: 本地节点索引序列，与本地节点ID序列一一对应。
            adj: 邻接矩阵，表示节点间的关系。

        """
        data = self.data[index]
        uid = np.array([data[0]], dtype=np.int)  # 用户ID
        u_input = data[1][-self.max_seq_len:]  # 用户浏览的视频ID序列
        u_input = list(reversed(u_input))  # 反转浏览序列
        max_n_node = self.max_seq_len
        u_input = u_input + (self.max_seq_len - len(u_input)) * [0]  # 填充序列至最大长度
        global_ids = np.array(u_input)  # 全局节点ID序列
        node = np.unique(u_input)  # 提取唯一节点
        local_nodes = node.tolist() + (max_n_node - len(node)) * [0]  # 本地节点ID序列，包含重复的节点
        adj = np.zeros((max_n_node, max_n_node))  # 邻接矩阵，初始化为全0
        for i in np.arange(len(u_input) - 1):
            u = np.where(node == u_input[i])[0][0]  # 当前节点在节点列表中的索引
            adj[u][u] = 1  # 自环，表示节点自身的关系
            if u_input[i + 1] == 0:  # 如果下一个节点为0，表示序列结束
                break
            v = np.where(node == u_input[i + 1])[0][0]  # 下一个节点在节点列表中的索引
            if u == v or adj[u][v] == 4:  # 如果当前节点和下一个节点相同或已经存在边连接，则跳过
                continue
            adj[v][v] = 1  # 自环，表示节点自身的关系
            if adj[v][u] == 2:  # 如果已经存在v到u的边，则将边的权重设置为4
                adj[u][v] = 4
                adj[v][u] = 4
            else:  # 如果不存在v到u的边，则将边的权重设置为2
                adj[u][v] = 2
                adj[v][u] = 3

        alias_inputs = [np.where(node == i)[0][0] for i in u_input]  # 获取本地节点索引

        seq_len = len(data[1][-self.max_seq_len:])  # 序列的实际长度
        mask = np.array([1 for i in range(seq_len)] + [0 for i in range(self.max_seq_len - seq_len)], dtype=np.int)  # 序列的掩码，用于指示有效元素

        global_ids = np.array(u_input)  # 全局节点ID序列
        local_ids = alias_inputs  # 本地节点索引序列

        seq_len = np.array(seq_len, dtype=np.int)  # 序列的实际长度
        label = np.array(data[2], dtype=np.int)  # 标签

        return uid, \
               global_ids, \
               mask, \
               seq_len, \
               label, \
               np.array(local_nodes, dtype=np.int), \
               np.array(local_ids, dtype=np.int), \
               adj  # 返回样本的各个特征值





def main():
    # Step 1: Load configuration from base.ini
    config = configparser.ConfigParser()
    config.read('../basic.ini')  # Make sure 'base.ini' is in the same directory
    config = config['default']

    # Step 2: Load train and test data
    data_path = "./dataset/"
    dataset = "lastfm"
    train_data, test_data, max_vid, max_uid = load_data(dataset, data_path)

    # Step 3: Create dataset objects
    train_dataset = SessionDataset(train_data, config, max_len=None)
    test_dataset = SessionDataset(test_data, config, max_len=None)

    # Step 4: Perform model training, evaluation, or inference
    # You can call your model training code here using train_dataset and test_dataset.

if __name__ == "__main__":
    main()
