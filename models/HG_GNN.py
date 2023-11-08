import dgl
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn.pytorch as dglnn
import torchsnooper
import pickle
import dgl.function as FN
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class HG_GNN(nn.Module):
    # @torchsnooper.snoop()
    def __init__(self, G, config, item_num, max_seq_len=10, max_sess=10):
        '''
        GNN模型的初始化函数。下面是对其中各部分的详细注释：

        super().__init__(): 调用父类的初始化函数。

        self.G = G.to(device): 将图数据移动到指定的设备上。

        self.max_sess = max_sess: 记录最大会话数。

        self.hidden_size = config['hidden_size']: 隐藏层的大小，从配置中获取。

        self.em_size = config['embed_size']: 嵌入向量的大小，从配置中获取。

        self.pos_embedding = nn.Embedding(200, self.em_size): 位置嵌入层，将位置信息映射为嵌入向量。

        self.v2e = nn.Embedding(G.number_of_nodes(), self.em_size).to(device): 节点嵌入层，将图中的节点映射为嵌入向量。这里使用了一个嵌入层来存储节点的嵌入向量，并将其移动到指定的设备上。

        self.conv1 = dglnn.SAGEConv(self.em_size, self.em_size, 'mean'): 第一层SAGE卷积，用于在图上进行信息传播和聚合。SAGE（GraphSAGE）是一种图卷积神经网络模型。

        dropout = config["dropout"]: 从配置中获取Dropout的概率。

        self.emb_dropout = nn.Dropout(p=dropout): 嵌入层的Dropout，用于随机丢弃嵌入向量的部分元素，以减少过拟合。

        self.gru = nn.GRU(self.em_size, self.hidden_size, 1): GRU层，用于对输入序列进行建模和提取序列特征。

        self.max_seq_len = max_seq_len: 记录最大序列长度。

        self.W = nn.Linear(self.em_size, self.em_size): 线性变换层，用于对嵌入向量进行线性变换。

        self.linear_one, self.linear_two, self.linear_three: 节点嵌入部分的线性变换层，用于将节点嵌入向量映射到另一个向量空间。

        self.a_1, self.a_2, self.v_t: GRU嵌入部分的线性变换层。

        self.ct_dropout = nn.Dropout(dropout): 上下文的Dropout，用于随机丢弃上下文向量的部分元素。

        self.user_transform: 用户变换模块，用于将用户嵌入向量映射到另一个向量空间。

        self.gru_transform: GRU变换模块，用于将GRU隐藏状态向量映射到另一个向量空间。

        self.sigmoid_concat: 用于将两个向量进行拼接，并通过Sigmoid函数输出一个标量。

        self.w_1, self.w_2, self.glu1, self.glu2: 用于进行门控线性单元（GLU）操作的参数。

        self.w_3, self.w_4, self.glu3, self.glu4: 用于进行门控线性单元（GLU）操作的参数。

        self.reset_parameters(): 初始化模型参数。

        self.item_num = item_num: 记录物品的数量。
        Args:
            G:
            config:
            item_num:
            max_seq_len:
            max_sess:
        '''
        super().__init__()
        self.G = G.to(device)  # 图数据，将其移动到指定设备上
        self.max_sess = max_sess  # 最大会话数

        self.hidden_size = config['hidden_size']  # 隐藏层的大小
        self.em_size = config['embed_size']  # 嵌入向量的大小
        self.pos_embedding = nn.Embedding(200, self.em_size)  # 位置嵌入层

        self.v2e = nn.Embedding(G.number_of_nodes(), self.em_size).to(device)  # 节点嵌入层

        self.conv1 = dglnn.SAGEConv(self.em_size, self.em_size, 'mean')  # 第一层SAGE卷积
        # self.conv2 = dglnn.SAGEConv(self.em_size, self.em_size, 'mean')  # 第二层SAGE卷积

        dropout = config["dropout"]  # Dropout概率
        self.emb_dropout = nn.Dropout(p=dropout)  # 嵌入层的Dropout
        self.gru = nn.GRU(self.em_size, self.hidden_size, 1)  # GRU层

        self.max_seq_len = max_seq_len  # 最大序列长度
        self.W = nn.Linear(self.em_size, self.em_size)  # 线性变换层

        # 节点嵌入
        self.linear_one = nn.Linear(self.em_size, self.em_size, bias=True)
        self.linear_two = nn.Linear(self.em_size, self.em_size, bias=True)
        self.linear_three = nn.Linear(self.em_size, 1, bias=False)

        # GRU嵌入
        self.a_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.a_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_t = nn.Linear(self.hidden_size, 1, bias=False)

        self.ct_dropout = nn.Dropout(dropout)  # 上下文Dropout

        self.user_transform = nn.Sequential(
            # nn.ReLU(),
            nn.Linear(self.em_size, self.em_size, bias=True)

            # nn.BatchNorm1d(predict_em_size, momentum=0.5),
        )

        self.gru_transform = nn.Sequential(
            # nn.ReLU(),
            nn.Linear(self.hidden_size * 2, self.em_size, bias=True)
            # nn.BatchNorm1d(predict_em_size, momentum=0.5),
        )

        self.sigmoid_concat = nn.Sequential(
            nn.Linear(self.em_size * 2, 1, bias=True),
            nn.Sigmoid()
            # nn.BatchNorm1d(predict_em_size, momentum=0.5),
        )

        self.w_1 = nn.Parameter(torch.Tensor(2 * self.em_size, self.em_size))
        self.w_2 = nn.Parameter(torch.Tensor(self.em_size, 1))
        self.glu1 = nn.Linear(self.em_size, self.em_size)
        self.glu2 = nn.Linear(self.em_size, self.em_size, bias=False)

        self.w_3 = nn.Parameter(torch.Tensor(self.em_size, self.em_size))
        self.w_4 = nn.Parameter(torch.Tensor(self.em_size, 1))
        self.glu3 = nn.Linear(self.em_size, self.em_size)
        self.glu4 = nn.Linear(self.em_size, self.em_size, bias=False)

        self.reset_parameters()

        self.item_num = item_num

    def reset_parameters(self):
        '''
            重置模型参数的方法。根据模型中em_size属性的值，计算标准差stdv，然后对模型的每个参数进行均匀分布的随机初始化，取值范围为[-stdv, stdv]。
        Returns:

        '''
        stdv = 1.0 / math.sqrt(self.em_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_hidden_vector(self, hidden, mask, pos_idx):
        """
        计算隐藏向量。

        参数:
            hidden: 输入的隐藏向量，形状为 [batch_size, length, embedding_size]。
            mask: 掩码，指示哪些位置是有效的，形状为 [batch_size, length]。
            pos_idx: 位置索引，用于获取位置嵌入向量，形状为 [batch_size, length]。

        返回值:
            select: 选择的隐藏向量，形状为 [batch_size, embedding_size]。
            tmp: 计算隐藏向量中的临时变量，形状为 [batch_size, embedding_size]。
        """
        mask = mask.float().unsqueeze(-1)  # 将掩码转换为浮点型并添加一个维度，形状变为 [batch_size, length, 1]
        batch_size = hidden.shape[0]  # 批次大小
        length = hidden.shape[1]  # 序列长度
        pos_emb = self.pos_embedding(pos_idx)  # 获取位置嵌入向量，形状为 [batch_size, length, embedding_size]
        tmp = torch.sum(hidden * mask, -2) / torch.sum(mask, 1)  # 计算隐藏向量中的临时变量，形状为 [batch_size, embedding_size]
        hs = tmp.unsqueeze(-2).repeat(1, length, 1)  # 复制临时变量，形状为 [batch_size, length, embedding_size]
        nh = torch.matmul(torch.cat([pos_emb, hidden], -1),
                          self.w_1)  # 将位置嵌入向量和隐藏向量拼接，并通过线性层处理，形状为 [batch_size, length, hidden_size]
        nh = torch.tanh(nh)  # 经过激活函数 tanh
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))  # 通过门控线性单元处理，形状不变
        beta = torch.matmul(nh, self.w_2)  # 通过线性层处理，形状为 [batch_size, length, embedding_size]
        beta = beta * mask  # 将掩码应用于 beta，形状不变
        select = torch.sum(beta * hidden, 1)  # 根据 beta 对隐藏向量进行加权求和，形状为 [batch_size, embedding_size]

        return select, tmp

    def sess_user_vector(self, user_vec, note_embeds, mask):
        """
        计算会话用户向量。

        参数:
            user_vec: 用户向量，形状为 [batch_size, 1, embedding_size]。
            note_embeds: 笔记嵌入向量，形状为 [batch_size, length, embedding_size]。
            mask: 掩码，指示哪些位置是有效的，形状为 [batch_size, length]。

        返回值:
            select: 选择的会话用户向量，形状为 [batch_size, embedding_size]。
        """
        mask = mask.float().unsqueeze(-1)  # 将掩码转换为浮点型并添加一个维度，形状变为 [batch_size, length, 1]
        hs = user_vec.repeat(1, mask.shape[1], 1)  # 将用户向量复制，形状为 [batch_size, length, embedding_size]
        nh = torch.matmul(note_embeds, self.w_3)  # 将笔记嵌入向量通过线性层处理，形状为 [batch_size, length, embedding_size]
        nh = torch.tanh(nh)  # 经过激活函数 tanh
        nh = torch.sigmoid(self.glu3(nh) + self.glu4(hs))  # 通过门控线性单元处理，形状不变
        beta = torch.matmul(nh, self.w_4)  # 通过线性层处理，形状为 [batch_size, length, embedding_size]
        beta = beta * mask  # 将掩码应用于 beta，形状不变
        select = torch.sum(beta * note_embeds, 1)  # 根据 beta 对笔记嵌入向量进行加权求和，形状为 [batch_size, embedding_size]

        return select

    def forward(self, user, seq, mask, seq_len, pos_idx):
        """
        前向传播方法。
        从            outputs = model(uid.to(device), browsed_ids.to(device), mask.to(device), seq_len.to(device), pos_idx.to(device))  # 前向传播计算模型输出
        传入
        参数:
            user: 用户索引，形状为 [batch_size, 1]。
            seq: 序列索引，形状为 [batch_size, length]。
            mask: 掩码，指示哪些位置是有效的，形状为 [batch_size, length]。
            seq_len: 序列的实际长度，形状为 [batch_size]。
            pos_idx: 位置索引，用于获取位置嵌入向量，形状为 [batch_size, length]。

        返回值:
            scores: 预测得分，形状为 [batch_size, item_num]。
        """
        user = user + self.item_num

        # HG-GNN

        # 修改参数的数据类型为长整型（long type） -- by Fuyy
        pos_idx = pos_idx.long()

        h1 = self.conv1(self.G, self.emb_dropout(self.v2e(torch.arange(0, self.G.number_of_nodes()).long().to(device))))
        #  conv1 = dglnn.SAGEConv(self.em_size, self.em_size, 'mean')
        h1 = F.relu(h1)

        bs = seq.size()[0]
        L = seq.size()[1]

        node_list = seq
        node_list = node_list.long()
        user = user.long()

        item_embeds = (h1[node_list] + self.v2e(node_list)) / 2
        user_embeds = (h1[user] + self.v2e(user)) / 2
        node_embeds = item_embeds.view((bs, L, -1))

        seq_embeds = user_embeds.squeeze(1)  # [batch_size, embedding_size]

        sess_vec, avg_sess = self.compute_hidden_vector(node_embeds, mask, pos_idx)  # 计算隐藏向量
        sess_user = self.sess_user_vector(user_embeds, node_embeds, mask)  # 计算会话用户向量

        alpha = self.sigmoid_concat(torch.cat([sess_vec, sess_user], 1))  # 使用门控单元融合隐藏向量和会话用户向量

        seq_embeds = seq_embeds + (alpha * sess_vec + (1 - alpha) * sess_user)  # 融合后的向量加权相加

        item_embs = self.v2e.weight[1:]  # 获取物品嵌入向量

        scores = torch.matmul(seq_embeds, item_embs.permute(1, 0))  # 计算预测得分

        return scores