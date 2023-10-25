import os
import pickle
import math
from operator import itemgetter

import dgl
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import numpy as np
import pandas as pd
# from pandarallel import pandarallel
import pickle
from tqdm import tqdm
from dgl import random, seed
from utils.config import Configurator

from utils.tools import get_time_dif, Logger
from data_processor.data_loader import load_data, SessionDataset
from build_graph import uui_graph, sample_relations
from data_processor.date_helper import LastFM_Process

import torch.nn.functional as F
from models import HG_GNN
import logging

# 指定配置文件
config_file = 'basic.ini'

# 从指定文件加载配置
conf = Configurator(config_file)

# 判断是否可用CUDA，并选择设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 打印正在加载的配置文件
print('从 %s 加载配置...' % config_file)

# 创建日志记录器
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

# 创建日志处理程序，将日志写入文件
handler = logging.FileHandler("./logs/log_{}_{}_{}.txt".format(conf['recommender'], \
                                                               conf['dataset.name'], conf['comment']))
handler.setLevel(logging.INFO)

# 设置日志格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 创建控制台日志处理程序
console = logging.StreamHandler()
console.setLevel(logging.INFO)

# 设置日志格式
handler.setFormatter(formatter)

# 将处理程序添加到日志记录器
logger.addHandler(handler)
logger.addHandler(console)

# 数据预处理
lp = LastFM_Process(conf)
lp._read_raw_data()
lp._split_data()

# 加载训练数据、测试数据，以及最大的视频ID和用户ID
train_data_ater, test_data, max_vid, max_uid = load_data(conf['dataset.name'], conf['dataset.path'])

# 打印当前数据集名称和相关统计信息
print('当前数据集:', conf['dataset.name'])
print('训练集大小:', len(train_data_ater))
# print('验证集大小:', len(val_data))
print('测试集大小:', len(test_data))
print('视频数量:', max_vid)

# 更新配置中的视频和用户数量
conf.change_attr('dataset.n_items', max_vid + 1)
conf.change_attr('dataset.n_users', max_uid + 1)

# 打印最大的视频和用户数量
print("视频数量 {} | 用户数量 {}".format(max_vid, max_uid))



def train(config, model, device, train_iter, test_iter=None):
    start_time = time.time()  # 记录训练开始的时间
    model.train()  # 设置模型为训练模式
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)  # 定义优化器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["lr_dc_step"], gamma=config["lr_dc"])  # 定义学习率衰减策略

    total_batch = 0  # 总的批次数
    dev_best_loss = float('inf')  # 最好的验证集损失
    last_improve = 0  # 记录上次验证集损失提升的轮数
    flag = False  # 标记是否提前停止训练
    AUC_best = 0  # 最好的AUC值
    loss_list = []  # 记录每个批次的训练损失
    Log = Logger(fn="./logs/log_{}_{}_{}.log".format(conf['recommender'], conf['dataset.name'], conf['comment']))  # 日志记录器
    best_acc = 0  # 最好的准确率
    batchs = train_iter.__len__()  # 训练数据的批次数

    for epoch in range(config['epoch']):  # 遍历每个训练轮次  20次
        epoch_t = time.time()  # 记录当前轮次开始的时间
        print('Epoch [{}/{}]'.format(epoch + 1, config['epoch']))  # 打印当前轮次的信息
        loss_records = []  # 记录每个批次的训练损失
        L = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
        for i, (uid, browsed_ids, mask, seq_len, label, pos_idx) in enumerate(train_iter):  # 遍历每个训练批次

            '''
            tensor([[ 6],
        [ 6],
            ...
        [ 6],
        [ 1],
        [ 4],
        [ 6],
        [ 3]], dtype=torch.int32)
tensor([[1322, 1335,   37,  ..., 1629, 1372,    0],
        [ 176, 1335, 1541,  ..., 1330, 1347, 1467],
        [ 356,  176,  364,  ...,  175,  419,    0],
        ...,
        [ 193,  976,  284,  ...,    0,    0,    0],
        [1361,    0,    0,  ...,    0,    0,    0],
        [ 879,  920,    0,  ...,    0,    0,    0]], dtype=torch.int32)
tensor([[1, 1, 1,  ..., 1, 1, 0],
        [1, 1, 1,  ..., 1, 1, 1],
            '''
            print(uid)
            print(browsed_ids)
            print(mask)
            print(seq_len)
            print(label)
            print(pos_idx)

            model.train()  # 设置模型为训练模式
            outputs = model(uid.to(device), browsed_ids.to(device), mask.to(device), seq_len.to(device), pos_idx.to(device))  # 前向传播计算模型输出
            model.zero_grad()  # 梯度清零
            loss = L(outputs, (label - 1).to(device).squeeze().long())  # 计算损失
            loss_list.append(loss.item())  # 记录当前批次的训练损失
            loss_records.append(loss.item())  # 记录当前批次的训练损失
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新模型参数
            STEP_SIZE = 1000  # 每隔STEP_SIZE个批次打印一次训练信息
            improve = '*'  # 当前批次是否有改进的标记

            if total_batch % STEP_SIZE == 0:  # 达到打印训练信息的条件
                time_dif = get_time_dif(start_time)  # 计算从训练开始到当前的时间差
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.6},  Time: {2} {3}'  # 训练信息的格式
                logger.info(msg.format(total_batch, np.mean(loss_list), time_dif, improve))  # 打印训练信息
                loss_list = []  # 清空当前批次的训练损失列表
            total_batch += 1  # 更新总的批次数

        runtime = f"\nepoch runtime : {time.time() - epoch_t:.2f}s\n"  # 当前轮次的运行时间
        logger.info(runtime)  # 打印当前轮次的运行时间

        print('preformance on test set....')  # 打印在测试集上的性能
        scheduler.step()  # 更新学习率
        acc, info = evaluate_topk(config, model, test_iter)  # 在测试集上评估模型性能

        if acc > best_acc:  # 如果当前准确率超过最好的准确率
            best_acc = acc
            msg = f'epoch[{epoch}] test :{info}'
            Log.log(msg, red=True)
            logger.info(msg)
            last_improve = 0  # 重置验证集损失提升的轮数
            if config.save_flag:  # 如果设置了保存模型的标志
                torch.save(model.state_dict(), config["save_path"] + '/{}_epoch{}_{}.ckpt'.format(config['recommender'], config['epoch'], config['comment']))  # 保存模型参数
        else:
            msg = f'epoch[{epoch}] test :{info}'
            ##Log.log(msg, red=False)
            logger.info(msg)
            last_improve += 1  # 更新验证集损失提升的轮数
            if last_improve >= config['patience']:  # 如果验证集损失提升的轮数超过了设置的耐心值
                logger.info('Early stop: No more improvement')  # 打印提前停止训练的信息
                break  # 停止训练


# 定义指标函数metrics，用于计算准确率、MRR和NDCG
def metrics(res, labels):
    res = np.concatenate(res)  # 将结果列表连接成一个数组
    acc_ar = (res == labels)  # 计算预测结果和真实标签是否相等 [BS, K]
    acc = acc_ar.sum(-1)  # 计算每个样本在top-K中预测正确的数量

    rank = np.argmax(acc_ar, -1) + 1  # 计算每个样本在top-K中的排名
    mrr = (acc / rank).mean()  # 计算平均倒数排名
    ndcg = (acc / np.log2(rank + 1)).mean()  # 计算归一化折损累计增益
    return acc.mean(), mrr, ndcg


# 定义评估函数evaluate_topk，用于评估推荐模型在top-K准确率、MRR和NDCG上的性能
def evaluate_topk(config, model, data_iter, K=20):
    model.eval()  # 将模型切换到评估模式
    hit = []  # 存储是否命中的列表
    res50 = []  # 存储top-50的推荐结果列表
    res20 = []  # 存储top-20的推荐结果列表
    res10 = []  # 存储top-10的推荐结果列表
    res5 = []  # 存储top-5的推荐结果列表
    mrr = []  # 存储MRR值的列表
    labels = []  # 存储真实标签的列表
    uids = []  # 存储用户ID的列表
    t0 = time.time()  # 记录开始时间
    with torch.no_grad():  # 关闭梯度计算
        with tqdm(total=(data_iter.__len__()), desc='Predicting', leave=False) as p:  # 使用进度条显示预测进度
            for i, (uid, browsed_ids, mask, seq_len, label, pos_idx) in (enumerate(data_iter)):  # 遍历数据迭代器
                # print(datas)
                outputs = model(uid.to(device), browsed_ids.to(device), mask.to(device), seq_len.to(device),
                                pos_idx.to(device)
                                # his_ids.to(device),
                                # his_mat.to(device),
                                # his_mask.to(device),
                                # his_seq_mask.to(device)
                                )  # 使用模型进行推断，得到输出结果
                sub_scores = outputs.topk(K)[1].cpu()  # 获取top-K的推荐结果
                res20.append(sub_scores)  # 将top-20的推荐结果存储到列表中
                res10.append(outputs.topk(10)[1].cpu())  # 将top-10的推荐结果存储到列表中
                res5.append(outputs.topk(5)[1].cpu())  # 将top-5的推荐结果存储到列表中
                res50.append(outputs.topk(50)[1].cpu())  # 将top-50的推荐结果存储到列表中
                labels.append(label)  # 将真实标签存储到列表中
                # uids.append(datas['user_id'])
                p.update(1)  # 更新进度条
    labels = np.concatenate(labels)  # 将列表中的标签连接成一个数组
    labels = labels - 1  # 将标签减一，使其从0开始计数
    acc50, mrr50, ndcg50 = metrics(res50, labels)  # 计算top-50的指标
    acc20, mrr20, ndcg20 = metrics(res20, labels)  # 计算top-20的指标
    acc10, mrr10, ndcg10 = metrics(res10, labels)  # 计算top-10的指标
    acc5, mrr5, ndcg5 = metrics(res5, labels)  # 计算top-5的指标

    print("Top20 : acc {} , mrr {}, ndcg {}".format(acc20, mrr20, ndcg20))  # 打印top-20的指标```python
    print("Top10 : acc {} , mrr {}, ndcg {}".format(acc10, mrr10, ndcg10))  # 打印top-10的指标
    print("Top5 : acc {} , mrr {}, ndcg {}".format(acc5, mrr5, ndcg5))  # 打印top-5的指标

    pred_time = time.time() - t0  # 计算推断时间
    # acc=acc.mean()
    msg = 'Top-{} acc:{:.3f}, mrr:{:.4f}, ndcg:{:.4f}, time: {:.1f}s \n'.format(20, acc20 * 100, mrr20 * 100,
                                                                                ndcg20 * 100, pred_time)  # 构建输出消息
    msg += 'Top-{} acc:{:.3f}, mrr:{:.4f}, ndcg:{:.4f} \n'.format(10, acc10 * 100, mrr10 * 100, ndcg10 * 100)  # 构建输出消息
    msg += 'Top-{} acc:{:.3f}, mrr:{:.4f}, ndcg:{:.4f} \n'.format(5, acc5 * 100, mrr5 * 100, ndcg5 * 100)  # 构建输出消息
    # msg += 'Top-{} acc:{:.3f}, mrr:{:.4f}, ndcg:{:.4f} \n'.format(50, acc50 * 100, mrr50 * 100, ndcg50 * 100)

    return acc20, msg  # 返回top-20的准确率和输出消息

if __name__ == "__main__":
    # 设置随机种子
    seed(430)  # 设置随机种子为430，用于实现可复现的随机性
    random.seed(430)
    torch.manual_seed(430)
    torch.cuda.manual_seed_all(430)
    torch.backends.cudnn.deterministic = True

    SZ = 12  # 样本大小
    SEQ_LEN = 10  # 序列长度

    # 对数据集进行采样
    sample_relations(conf['dataset.name'], conf['dataset.n_items'], sample_size=SZ)  # 根据指定的数据集名称和物品数量，采样样本大小为SZ

    # 对图进行操作，获取图g和物品数量item_num
    g, item_num = uui_graph(conf['dataset.name'], sample_size=SZ, topK=20, add_u=False, add_v=False)
    # 通过使用指定的数据集名称和采样大小SZ，从图中获取图g和物品数量item_num
    # topK表示保留每个节点的最高K个邻居节点，add_u和add_v表示是否添加边为(u, v)的逆边和(v, u)的逆边






    print(g)  # 打印图g








    # 测试01 begin---------加载子图数据---2023/10/24--20：00
    # graphs, _ = dgl.load_graphs(r"E:\MyCode\PycharmCode\DGSR\Data\Games_graph.bin")
    # trainGraphs, _t = dgl.load_graphs(r"E:\MyCode\PycharmCode\DGSR\Newdata\Games_50_50_2\train\0\0_1.bin")
    # graph = graphs[0]  # 提取第一个图
    # trainGraph = trainGraphs[0]  # 提取第一个图

    # 测试01---end




    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 判断可用设备，如果有CUDA则使用GPU，否则使用CPU

    # train_data是list  [[1, [1, 2, 3], [4]],[1, [1, 2, 3, 4, 5, 6, 7, 8, 9], [10]]]   这种形式
    train_data = SessionDataset(train_data_ater, conf, max_len=SEQ_LEN)
    # 目前来看，只是将train_data封装成SessionDataset形式，{data,max_seq_len-->手动设置}

    # 创建训练数据集，使用train_data和配置conf，设置最大序列长度为10
    # train_data表示训练数据，conf表示配置信息，max_len表示序列的最大长度

    test_data = SessionDataset(test_data, conf, max_len=SEQ_LEN)
    # 创建测试数据集，使用test_data和配置conf，设置最大序列长度为10
    # test_data表示测试数据，conf表示配置信息，max_len表示序列的最大长度

    train_iter = DataLoader(dataset=train_data,
                            batch_size=conf["batch_size"],
                            num_workers=4,
                            drop_last=False,
                            shuffle=True,
                            pin_memory=False)
    # 创建训练数据迭代器
    # dataset表示要迭代的数据集，batch_size表示每个批次的样本数量，num_workers表示用于数据加载的子进程数量
    # drop_last表示是否丢弃最后一个批次（如果批次大小不完整），shuffle表示是否对数据进行随机洗牌，pin_memory表示是否将数据加载到固定内存中

    test_iter = DataLoader(dataset=test_data,
                           batch_size=conf["batch_size"] * 16,
                           num_workers=4,
                           drop_last=False,
                           shuffle=False,
                           pin_memory=False)
    # 创建测试数据迭代器
    # dataset表示要迭代的数据集，batch_size表示每个批次的样本数量，num_workers表示用于数据加载的子进程数量
    # drop_last表示是否丢弃最后一个批次（如果批次大小不完整），shuffle表示是否对数据进行随机洗牌，pin_memory表示是否将数据加载到固定内存中

    # model = HG_GNN(g, conf, item_num, SEQ_LEN).to(device)


    # 测试1---更改图为DG图
    model = HG_GNN(g, conf, item_num, SEQ_LEN).to(device)
    # 创建模型实例
    # g表示图数据，conf表示配置信息，item_num表示物品数量，SEQ_LEN表示序列长度
    # 将模型移动到设备(device)，使用GPU加速计算（如果可用）

    train(conf, model, device, train_iter, test_iter)
    # 进行训练
    # conf表示配置信息，model表示模型，device表示设备，train_iter表示训练数据迭代器，test_iter表示测试数据迭代器