"""
data_helper.py

    preprocess the raw datasets.(LastFM)
"""
import os
import time
import pickle
import operator
import numpy as np
import pandas as pd
from pandas import Timedelta
from tqdm import tqdm

# 使用配置文件类导入配置
# from utils.config import Configurator
import configparser


class BasicGraph:
    def __init__(self, min_cnt=1):
        self.min_cnt = min_cnt
        self.edge_cnt = {}
        self.adj = {}
        self._nb_edges = 0 # edges nums

    def add_edge(self, a, b):
        e = (a, b)
        self.edge_cnt.setdefault(e, 0)
        self.edge_cnt[e] += 1
        # first appear
        if self.edge_cnt[e] == self.min_cnt:
            self.adj.setdefault(a, [])
            self.adj[a].append(b)
            self._nb_edges += 1

    def has_edge(self, a, b):
        cnt = self.edge_cnt.get((a, b), 0)
        return cnt >= self.min_cnt

    def get_edges(self):
        edges = sorted([(a, b) for (a, b), cnt in self.edge_cnt.items() if cnt >= self.min_cnt])
        return edges

    def get_adj(self, a):
        return self.adj.get(a, [])

    def nb_edges(self):
        return self._nb_edges


class Data_Process(object):
    def __init__(self,config):

        self.dataset=config['dataset.name']
        self.data_path=config['dataset.path']

        self.conf=config
        self.vid_map={}
        self.filter_vid={}

    def _yoochoose_item_attr(self,data_home='./raw_data/yoochoose',saved_path='./dataset'):
        category_dict={}

        # df=pd.read_csv(f'{data_home}/yoochoose-clicks.dat',header=None,names=["sid", "Timestamp", "vid", "Category"],usecols=["vid", "Category"])
        df=pd.read_csv(f'E:/MyCode/PycharmCode/HG-GNN/data_processor/raw_data/yoochoose/yoochoose-clicks.dat',header=None,names=["sid", "Timestamp", "vid", "Category"],usecols=["vid", "Category"])
        print(df)
        print("aaaaaaaaadf")
        new_cate=df.drop_duplicates(['vid','Category'])
        new_cate=new_cate[new_cate['Category'].isin([str(_) for _ in range(1,13)])]#['Category']#.value_counts()
        vid=new_cate['vid'].tolist()
        cate=new_cate['Category'].tolist()
        category_dict=dict(zip(vid,cate))
        with open(f'{saved_path}/yc_cate.pkl', 'wb') as f:
            pickle.dump(category_dict,f)


    def _yoochoose_(self,frac = 4,data_home='./raw_data/yoochoose',saved_path='./dataset'):
        """
        handle with the raw data format of yoochoose.
        """
        sid2vid_list = {}
        # with open(f'{data_home}/yoochoose-buys.dat', 'r') as f:
        with open(f'E:/MyCode/PycharmCode/HG-GNN/data_processor/raw_data/yoochoose/yoochoose-buys.dat', 'r') as f:
            for line in tqdm(f,desc='read buy',leave=True):
                #pbar.update(1)
                #cnt += 1
                #if N > 0 and cnt > N: break
                line = line[:-1]
                sid, ts, vid, _, _ = line.split(',')
                ts = int(time.mktime(time.strptime(ts[:19], '%Y-%m-%dT%H:%M:%S')))
                sid2vid_list.setdefault(sid, [])
                sid2vid_list[sid].append([vid, 0, ts])

        # session_id,timestamp,item_id,category
        # 1,2014-04-07T10:51:09.277Z,214536502,0
        cnt = 0
        # with open(f'{data_home}/yoochoose-clicks.dat', 'r') as f:
        with open(f'E:/MyCode/PycharmCode/HG-GNN/data_processor/raw_data/yoochoose/yoochoose-clicks.dat', 'r') as f:
            for line in tqdm(f,desc='read clicks',leave=True):
                #pbar.update(1)
                cnt += 1
                #if N > 0 and cnt > N: break
                line = line[:-1]
                sid, ts, vid, cate = line.split(',')
                ts = int(time.mktime(time.strptime(ts[:19], '%Y-%m-%dT%H:%M:%S')))

                sid2vid_list.setdefault(sid, [])
                sid2vid_list[sid].append([vid, 1, ts])

        for sid in sid2vid_list:
            sid2vid_list[sid] = sorted(sid2vid_list[sid], key=lambda x: x[-1])

        n = len(sid2vid_list)
        # sort all sessions by the last time of the session
        yc = sorted(sid2vid_list.items(), key=lambda x: x[1][-1][-1])

        n_part = n // frac
        print('sid2vid len:',n)
        print('n_part:',n_part)
        yc_part = yc[-n_part:]

        out_path = f'yc_1_{frac}'
        os.mkdir(f'{saved_path}/{out_path}')
        with open(f'{saved_path}/{out_path}/data.txt', 'w') as f:
            for sid, vid_list in yc_part:
                vid_list = ','.join(map(lambda vid: ':'.join(map(str, vid)), vid_list))
                sess = ' '.join([sid, vid_list])
                f.write(sess + '\n')

    """
        def _read_diginetica(self,data_home='./raw_data/yoochoose',saved_path='./dataset'):
            diginetica raw_data:
            保留具有userid的数据
            view_df=pd.read_csv(data_home+'/train-item-views.csv',sep=';') #sessionId	userId	itemId	timeframe	eventdate
            view_df=view_df.dropna().reset_index(drop=True)
            view_df['eventdate']=view_df['eventdate'].apply(lambda x : time.mktime(time.strptime(x, '%Y-%m-%d')))

            return 
    """
    def _read_raw_data(self,dataset_name):
        """
        read raw data to select the target information.

        data format: session_id,[(vid,v_type)],session_end_time
        """
        cold_session=self.conf['dataset.filter.cold_session']
        cold_item=self.conf['dataset.filter.cold_item']

        vid_count={}
        data_list=[]

        with open('E:/MyCode/PycharmCode/HG-GNN/data_processor/dataset/lastfm/data.txt','r') as f:
        # with open(os.path.join(self.data_path,dataset_name)+'/data.txt','r') as f:
            for line in tqdm(f,desc='loading data.txt',leave=True):
                line = line[:-1]
                sid, vid_list_str = line.split()
                vid_list = []
                max_ts=0
                vid_list={}
                if len(vid_list_str.split(','))<=cold_session:
                    continue
                for vid in vid_list_str.split(','):

                    vid, cls, ts = vid.split(':')
                    #cls = int(cls)  # cls: 0, 1, 2, ...
                    if cls=='1':
                        if vid not in vid_count :
                            vid_count[vid]=1
                        else:
                            vid_count[vid]+=1
                    if int(ts)>max_ts:
                        max_ts = int(ts)
                    vid_list.setdefault(cls,[])
                    vid_list[cls].append(vid)
                if len(vid_list['1'])<=cold_session:
                    continue
                data_list.append([vid,vid_list,max_ts])

                    #vid_list.append([vid, cls, ts]) # item_id , behavior_type , timestamp
        # sort by vid appears
        sorted_counts = sorted(vid_count.items(), key=operator.itemgetter(1))

        filtered_data_list=[]
        for s in tqdm(data_list,desc='filterd sessions & items',leave=False):
            #print(s[1]) 0:buy 1:click
            filseq = list(filter(lambda i: vid_count[i] >= cold_item, s[1]['1']))
            if len(filseq) < 2:
                continue
            filtered_data_list.append(s)
        return filtered_data_list


    def _re_vindex(self,data_list,filter_flag):
        """
        reindex the items id starting from 1.

        params：
             filter_flag(bool): filter out the items not appear in the train set.

        return: only includes the vid lists

        """

        new_id=1
        new_data_list=[]
        for s in data_list:
            outseq=[]
            for vid in s[1]['1']:
                if filter_flag:
                    if vid not in self.vid_map:
                        self.filter_vid.setdefault(vid,0)
                        self.filter_vid[vid]+=1
                        continue

                if vid not in self.vid_map:
                    outseq.append(new_id)
                    self.vid_map[vid]=new_id
                    new_id+=1
                else:
                    outseq.append(self.vid_map[vid])
            new_data_list.append(outseq)

        #print(new_data_list[:10])
        return new_data_list


    def _split_data(self):
        """
        split the full dataset to train/val/test dataset.

        """
        splitter = self.conf['dataset.split']
        val_ratio = self.conf['dataset.val_ratio']

        data_list=self._read_raw_data(self.dataset)
        category_dict={}
       # print(self.data_path)
        #print(os.path.join(self.data_path,'yc_cate.pkl'))
        # with open(os.path.join(self.data_path,'yc_cate.pkl'),'rb') as f:
        #     raw_categ_dict=pickle.load(f)

        max_time=data_list[-1][-1]
        if splitter=='by_day':
            test_lens=self.conf['dataset.test_days']
            splitdate = max_time - test_lens*24*60*60
            train_sess = filter(lambda x: x[2] < splitdate, data_list)
            tes_sess = filter(lambda x: x[2] > splitdate, data_list)
            # reindex the item id
            train_sess=self._re_vindex(train_sess,filter_flag=False)
            tes_sess=self._re_vindex(tes_sess,filter_flag=True)

            val_lens= int(len(tes_sess)*val_ratio)

            val_sess=tes_sess[:val_lens]
            test_sess=tes_sess[val_lens:]
            # for key in tqdm(self.vid_map):
            #     try:
            #         category_dict[self.vid_map[key]]=raw_categ_dict[str(key)]
            #     except:
            #         print(key)


            print("session nums: train/val/test: ",len(train_sess),len(val_sess),len(tes_sess))

            print(train_sess[-3:],test_sess[-3:])

            print('item num:',len(self.vid_map))

            print('filtered item num:',len(self.filter_vid))

            train_sess={0:train_sess}
            val_sess={0:val_sess}
            test_sess={0:test_sess}
            # store the dataset with pickle
            # [vid_list_0,vid_list_1] (different behaviors,dict format)
            path = os.path.join(self.data_path,self.dataset)
            print(path)


            with open('E:/MyCode/PycharmCode/HG-GNN/data_processor/dataset/lastfm/train.pkl','wb') as f:
                pickle.dump(train_sess,f)

            with open('E:/MyCode/PycharmCode/HG-GNN/data_processor/dataset/lastfm/val.pkl','wb') as f:
                pickle.dump(val_sess,f)

            with open('E:/MyCode/PycharmCode/HG-GNN/data_processor/dataset/lastfm/test.pkl','wb') as f:
                pickle.dump(test_sess,f)

            # with open(os.path.join(self.data_path,self.dataset)+'/cate_dict.pkl','wb') as f:
            #     pickle.dump(category_dict,f)

    def _get_sample_adj(self,G):
        random_seed=self.conf['random_seed']
        adj_size = self.conf['graph.adj_size']
        N = self.conf['dataset.n_items']
        rdm = np.random.RandomState(random_seed)
        adj=[]
        w=[]
        adj.append([0]*adj_size)
        w.append([0]*adj_size)
        for node in tqdm(range(1, N),total=N - 1, desc='building adj',leave=False):
            #pbar.update(1)
            adj_list = G.get_adj(node) # get the adjacent nodes (M nodes)
            if len(adj_list) > adj_size:
                adj_list = rdm.choice(adj_list, size=adj_size, replace=False).tolist()
            mask = [0] * (adj_size - len(adj_list))
            adj_list = adj_list[:] + mask # set the masks for padding
            adj.append(adj_list)
            w_list = [G.edge_cnt.get((node, x), 0) for x in adj_list] # get the edge weight for each adj node of the target node.
            w.append(w_list)
        return [adj,w]


    # def _build_graph(self):
    #     """"
    #     follow the specific strategy to build the graph of the sessions.

    #     return adj data.

    #     """
    #     with open(os.path.join(self.data_path,self.dataset)+'/train.pkl','rb') as f:
    #         train_data_list=pickle.load(f)

    #     print('build the global weighed & directed graph structure')
    #     G_in=BasicGraph()
    #     G_out=BasicGraph()
    #     for sess in tqdm(train_data_list,'build the graph',leave=False):
    #         for i,vid in enumerate(sess['1']):
    #             if i==0:
    #                 continue
    #             now_node=vid
    #             pre_node=sess['1'][i-1]
    #             if now_node!=pre_node:
    #                 G_out.add_edge(pre_node,now_node)# out degree
    #                 G_in.add_edge(now_node,pre_node) # in degree

    #     adj0=self._get_sample_adj(G_in)
    #     adj1=self._get_sample_adj(G_in)
    #     with open(os.path.join(self.data_path,self.dataset)+'/adj.pkl','wb') as f:
    #         pickle.dump([adj0,adj1],f)


# 加载lastfm数据集的类

class LastFM_Process(Data_Process):
    def __init__(self, config):
        """
        初始化函数，用于创建LastFM_Process对象。

        参数：
        - config: 配置对象，包含相关参数的配置信息

        """

        '''
            先跳到Data_Process，加载到Data_Process的实例中
        '''
        super(LastFM_Process, self).__init__(config)
        self.interval = Timedelta(hours=8)
        self.full_df = None
        # 设置时间间隔和全局数据框为空

    def _update_id(self, df, field):
        """
        更新指定字段的ID值，将其转换为连续的整数编码。

        参数：
        - df: DataFrame，包含待更新的数据框
        - field: 字符串，待更新的字段名

        返回值：
        - df: DataFrame，更新后的数据框

        """
        labels = pd.factorize(df[field])[0]
        kwargs = {field: labels}
        df = df.assign(**kwargs)
        # 使用pd.factorize()函数将字段的取值转换为连续的整数编码
        # 将更新后的字段作为新的列添加到数据框中
        return df  # 返回更新后的数据框

    def _group_sessions(self, df):
        """
        将交互数据按照会话进行分组，生成会话ID。  如是同一用户且相差小于8小时  同1组 interval控制

        参数：
        - df: DataFrame，包含交互数据的数据框

        返回值：
        - df: DataFrame，添加了会话ID字段的数据框

        """
        df_prev = df.shift()
        is_new_session = (df.userId != df_prev.userId) | (
                df.timestamp - df_prev.timestamp > self.interval
        )
        session_id = is_new_session.cumsum() - 1
        # 判断当前交互是否属于新的会话，生成布尔序列
        # 使用cumsum()函数对布尔序列进行累加，生成会话ID

        df = df.assign(sessionId=session_id)
        # 将会话ID作为新的列添加到数据框中
        return df  # 返回添加了会话ID字段的数据框

    def remove_immediate_repeats(self, df):
        """
        去除相邻的重复交互记录。

        参数：
        - df: DataFrame，包含交互数据的数据框

        返回值：
        - df_no_repeat: DataFrame，去除相邻重复交互记录后的数据框

        """
        df_prev = df.shift()
        is_not_repeat = (df.sessionId != df_prev.sessionId) | (df.itemId != df_prev.itemId)
        df_no_repeat = df[is_not_repeat]
        # 判断当前交互是否与前一个交互重复，生成布尔序列
        # 根据布尔序列筛选出不重复的交互记录

        return df_no_repeat  # 返回去除相邻重复交互记录后的数据框

    def truncate_long_sessions(self, df, max_len=20, is_sorted=False):
        """
        截断过长的会话，使其最大长度不超过指定值。

        参数：
        - df: DataFrame，包含交互数据的数据框
        - max_len: 整数，会话的最大长度，默认为20
        - is_sorted: 布尔值，指示数据框是否已按会话ID和时间戳排序，默认为False

        返回值：
        - df_t: DataFrame，截断后的数据框，会话长度不超过指定值

        """
        if not is_sorted:
            df = df.sort_values(['sessionId', 'timestamp'])
        itemIdx = df.groupby('sessionId').cumcount()
        df_t = df[itemIdx < max_len]
        # 如果数据框未按会话ID和时间戳排序，则进行排序
        # 使用groupby().cumcount()函数生成每个交互在会话中的序号
        # 根据序号筛选出会话长度不超过指定值的交互记录

        return df_t  # 返回截断后的数据框

    def keep_top_n_items(self, df, n):
        """
        保留出现频次最高的前n个物品。

        参数：
        - df: DataFrame，包含交互数据的数据框
        - n: 整数，保留的物品数量

        返回值：
        - df_top: DataFrame，保留了出现频次最高的前n个物品的数据框

        """
        item_support = df.groupby('itemId', sort=False).size()
        top_items = item_support.nlargest(n).index
        df_top = df[df.itemId.isin(top_items)]
        # 统计物品的出现频次，并找到出现频次最高的前n个物品
        # 根据物品ID筛选出出现频次最高的物品的交互记录

        return df_top  # 返回保留了出现频次最高的前n个物品的数据框

    def filter_short_sessions(self, df, min_len=2):
        """
        过滤掉长度过短的会话。

        参数：
        - df: DataFrame，包含交互数据的数据框
        - min_len: 整数，会话的最小长度，默认为2

        返回值：
        - df_long: DataFrame，过滤掉长度过短的会话后的数据框

        """
        session_len = df.groupby('sessionId', sort=False).size()
        long_sessions = session_len[session_len >= min_len].index
        df_long = df[df.sessionId.isin(long_sessions)]
        # 统计会话的长度，并找到长度不小于指定值的会话
        # 根据会话ID筛选出长度不小于指定值的会话的交互记录

        return df_long  # 返回过滤掉长度过短的会话后的数据框

    def filter_infreq_items(self, df, min_support=5):
        """
        过滤掉出现频次过低的物品。

        参数：
        - df: DataFrame，包含交互数据的数据框
        - min_support: 整数，物品的最小支持度，默认为5

        返回值：
        - df_freq: DataFrame，过滤掉出现频次过低的物品后的数据框

        """
        item_support = df.groupby('itemId', sort=False).size()
        freq_items = item_support[item_support >= min_support].index
        df_freq = df[df.itemId.isin(freq_items)]
        # 统计物品的出现频次，并找到出现频次不小于指定值的物品
        # 根据物品ID筛选出出现频次不小于指定值的物品的交互记录

        return df_freq  # 返回过滤掉出现频次过低的物品后的数据框

    def filter_until_all_long_and_freq(self, df, min_len=2, min_support=5):
        """
        反复过滤数据框，直到所有会话都达到指定的最小长度，并且所有物品都达到指定的最小支持度。

        参数：
        - df: DataFrame，包含交互数据的数据框
        - min_len: 整数，会话的最小长度，默认为2
        - min_support: 整数，物品的最小支持度，默认为5

        返回值：
        - df: DataFrame，过滤后满足条件的数据框

        """
        while True:
            df_long = self.filter_short_sessions(df, min_len)
            df_freq = self.filter_infreq_items(df_long, min_support)
            if len(df_freq) == len(df):
                break
            df = df_freq
        # 反复调用filter_short_sessions()和filter_infreq_items()函数，直到不再过滤数据框
        # 如果每次过滤后的数据框长度与初始数据框长度相同，则停止过滤

        return df  # 返回满足条件的数据框


    def _agg_df(self, df):
        """
        对数据进行聚合操作，将每个用户的会话序列按照用户ID进行分组，并将每个会话中的物品ID序列聚合为列表形式。

        参数：
        - df: DataFrame，包含用户ID（userId）、会话ID（sessionId）和物品ID（itemId）的数据框

        返回值：
        - res: 字典，键为用户ID，值为聚合后的会话序列列表

        """
        res = {}  # 存储聚合后的会话序列字典
        for u, ug in tqdm(df.groupby('userId')):
            # 按照用户ID进行分组，ug为每个用户对应的DataFrame子集
            res.setdefault(u, [])
            # 为每个用户ID设置一个空列表作为默认值
            res[u] = ug.groupby('sessionId')['itemId'].agg(list).tolist()
            # 将每个会话中的物品ID序列聚合为列表形式，并将结果存储到字典res中

        return res  # 返回聚合后的会话序列字典




    def _agg_all_seq(self, df):
        """
        对数据进行聚合操作，将每个用户的会话序列按照用户ID进行分组，并将每个会话中的物品ID序列聚合为列表形式。

        参数：
        - df: DataFrame，包含用户ID（userId）、会话ID（sessionId）和物品ID（itemId）的数据框

        返回值：
        - res: 列表，包含每个用户的聚合后的会话序列列表

        """
        res = []  # 存储聚合后的会话序列列表
        for u, ug in tqdm(df.groupby('userId')):
            # 按照用户ID进行分组，ug为每个用户对应的DataFrame子集
            res += ug.groupby('sessionId')['itemId'].agg(list).tolist()
            # 将每个会话中的物品ID序列聚合为列表形式，并添加到res列表中
        with open('E:/MyCode/PycharmCode/HG-GNN/data_processor/dataset/lastfm/all_train_seq.txt','wb') as f:
        # with open(os.path.join(self.data_path,self.dataset)+'/all_train_seq.txt','wb') as f:
        # 将聚合后的会话序列列表保存为pkl文件
            pickle.dump(res, f)

        return res  # 返回聚合后的会话序列列表
    def _split_data(self):
        print('split data...')
        # 获取配置参数
        splitter = self.conf['dataset.split']  # 数据集拆分器
        val_ratio = self.conf['dataset.val_ratio']  # 验证集比例

        # 将字符类型的val_ratio转换为浮点数
        val_ratio = float(val_ratio)

        test_split = 0.2  # 测试集比例

        # 改成绝对路径
        df=pd.read_csv('E:/MyCode/PycharmCode/HG-GNN/data_processor/dataset/lastfm/data.csv',header=None,names=['userId', 'timestamp', 'itemId','sessionId'])
        # df=pd.read_csv(os.path.join(self.data_path,self.dataset)+'/data.txt',header=None,names=['userId', 'timestamp', 'itemId','sessionId'])
        print(df['userId'].nunique())
        print(df['itemId'].nunique())
        print(df['itemId'].max(),df['itemId'].min())

        # 根据sessionId对数据进行分组，并获取每个组的最大时间戳
        endtime = df.groupby('sessionId', sort=False).timestamp.max()
        endtime = endtime.sort_values()

        # 计算测试集的数量

        # 计算测试集的数量。endtime是一个包含会话结束时间的索引，len(endtime)表示总会话数。test_split是测试集的比例（0到1之间的浮点数），通过将总会话数与测试集比例相乘得到测试集的数量，使用int函数将结果转换为整数。
        num_tests = int(len(endtime) * test_split)
        # 根据计算得到的测试集数量，获取最后num_tests个会话的索引。这些会话的索引将被用作测试集的标识。
        test_session_ids = endtime.index[-num_tests:]

        # 将数据分为训练集和测试集

        # 通过筛选出不包含在测试集中的会话，得到训练集的DataFrame。df.sessionId.isin(test_session_ids)用于判断每个会话的ID是否在测试集的索引中，~表示取反操作，即保留不在测试集索引中的会话。
        df_train = df[~df.sessionId.isin(test_session_ids)]
        # 通过筛选出包含在测试集中的会话，得到测试集的DataFrame。df.sessionId.isin(test_session_ids)用于判断每个会话的ID是否在测试集的索引中，保留在测试集索引中的会话。reset_index(drop=True)用于重置测试集数据的索引，确保索引从0开始连续递增。
        df_test = df[df.sessionId.isin(test_session_ids)].reset_index(drop=True)

        # 重新映射测试集中的itemId，只保留在训练集中出现过的项
        df_test = df_test[df_test.itemId.isin(df_train.itemId.unique())]
        df_test = self.filter_short_sessions(df_test)

        # 重新映射训练集和测试集中的itemId，并将userId和itemId的值加1
        train_itemId_new, uniques = pd.factorize(df_train.itemId)
        df_train = df_train.assign(itemId=train_itemId_new)
        oid2nid = {oid: i for i, oid in enumerate(uniques)}
        test_itemId_new = df_test.itemId.map(oid2nid)
        df_test = df_test.assign(itemId=test_itemId_new)
        df_train['userId'] += 1
        df_train['itemId'] += 1
        df_test['userId'] += 1
        df_test['itemId'] += 1

        # 对训练集进行聚合处理
        self._agg_all_seq(df_train)

        # 打印训练集中userId的最小值和最大值，itemId的最大值和最小值，以及测试集中itemId的最大值和最小值
        print("df_train['userId'].min(),df_train['userId'].max()")
        print(df_train['userId'].min(), df_train['userId'].max())
        print("df_train['itemId'].max(),df_train['itemId'].min()")
        print(df_train['itemId'].max(), df_train['itemId'].min())
        print("df_test['itemId'].max(),df_test['itemId'].min()")
        print(df_test['itemId'].max(), df_test['itemId'].min())

        # 重置测试集的索引，并根据val_ratio从测试集中随机抽取一部分作为验证集
        df_test = df_test.reset_index(drop=True)
        df_val = df_test.sample(frac=val_ratio)
        part_test = df_test[~df_test.index.isin(df_val.index)]
        print(os.path.join(self.data_path,self.dataset)+"************")

        # 将处理后的训练集、验证集、测试集以及完整的测试集保存为pickle文件
        # with open(os.path.join(self.data_path,self.dataset)+'/train.pkl','wb') as f:
        # 这里使用`pickle.dump()`函数将经过聚合处理的训练集`df_train`保存为`train.pkl`文件。
        with open('E:/MyCode/PycharmCode/HG-GNN/data_processor/dataset/lastfm/train.pkl','wb') as f:
            pickle.dump(self._agg_df(df_train),f)        # 保存训练集

        # with open(os.path.join(self.data_path,self.dataset)+'/val.pkl','wb') as f:
        # 使用`pickle.dump()`函数将经过聚合处理的验证集`df_val`保存为`val.pkl`文件。
        with open('E:/MyCode/PycharmCode/HG-GNN/data_processor/dataset/lastfm/val.pkl','wb') as f:
            pickle.dump(self._agg_df(df_val),f)          # 保存验证集

        # with open(os.path.join(self.data_path,self.dataset)+'/test.pkl','wb') as f:
        # 使用`pickle.dump()`函数将经过聚合处理的测试集`part_test`保存为`test.pkl`文件。
        with open('E:/MyCode/PycharmCode/HG-GNN/data_processor/dataset/lastfm/test.pkl','wb') as f:
            pickle.dump(self._agg_df(part_test),f)

        # with open(os.path.join(self.data_path,self.dataset)+'/all_test.pkl','wb') as f:
        # 使用`pickle.dump()`函数将经过聚合处理的完整测试集`df_test`保存为`all_test.pkl`文件。
        with open('E:/MyCode/PycharmCode/HG-GNN/data_processor/dataset/lastfm/all_test.pkl','wb') as f:
            pickle.dump(self._agg_df(df_test),f)

    # 读取lastFM数据集文件的方法



    def _read_raw_data(self,topK=40000):

        '''
            读取文件，处理成只剩userID itemID timesTemp 三列，处理一些空白数据，分组，列重新编码
        Args:
            topK:
            保留的商品数量的上限，默认为40000。
        Returns:

        '''

        data_home='./raw_data/lastfm-1K'
        saved_path='./dataset'
        # csv_file=data_home+'/userid-timestamp-artid-artname-traid-traname.tsv'
        csv_file='E:/MyCode/PycharmCode/HG-GNN/data_processor/raw_data/lastfm-1K/userid-timestamp-artid-artname-traid-traname.tsv'
        print(f'read raw data from {csv_file}')

        # df: 包含LastFM数据的DataFrame对象。
        # DataFrame对象是pandas库中的一种数据结构，用于存储和操作二维数据。它类似于电子表格或数据库表格，由行和列组成。
        # pd--pandas
        df = pd.read_csv(
            csv_file,
            sep='\t',
            header=None,
            names=['userId', 'timestamp', 'itemId','artname','traid','traname'],# 参数指定了为DataFrame中的列分配的名称。这些列名依次表示用户ID、时间戳、物品ID、艺术家名称、曲目ID和曲目名称。
            usecols=['userId', 'timestamp', 'itemId'],# 参数指定了在DataFrame中保留的列。在这个例子中，只保留了用户ID、时间戳和物品ID这三列，其他列将被忽略。
            parse_dates=['timestamp'],# 参数将'timestamp'列解析为日期时间格式。这样可以方便地对时间戳进行处理和分析。
            infer_datetime_format=True,
        )

        print('start preprocessing')
        print(df.head())

        #  删除DataFrame中含有缺失值（NaN）的行，确保数据的完整性。
        df = df.dropna()
        print("dropna后")
        print(df.head())

        # 对 df 中的 'userId' 列重新编制索引，从1开始
        df = self._update_id(df,'userId')   # 将userID和ItemID因子化 将分类或离散变量转换为整数编码     pandas库中的factorize函数来对DataFrame中的某一列进行因子化（Factorization）操作。
        print("_update_id后")
        print(df.head())

        # 对 df 中的 'itemId' 列重新编制索引，从1开始
        df = self._update_id(df, 'itemId')

        print("_update_id后")
        print(df.head())
        # 根据 'userId' 和 'timestamp' 列的值对DataFrame进行排序，按照用户ID和时间戳进行升序排序。
        df = df.sort_values(['userId', 'timestamp'])
        print("sort_values")
        print(df.head())
        # 根据 'userId' 和 'timestamp' 列对会话进行分组  如是同一用户且相差小于8小时  同1组 interval控制
        df = self._group_sessions(df)
        print(df.head(30))

        # 移除会话中连续出现的相同商品项
        '''
            这一步是否可以搞---重复数据说明更喜欢这一类？？
        '''
        df = self.remove_immediate_repeats(df)
        print(df.head(30))

        # 截断过长的会话，使其不超过指定长度
        df = self.truncate_long_sessions(df, is_sorted=True)
        print(df.head(30))

        # 保留出现次数最多的前 topK 个商品
        df = self.keep_top_n_items(df, topK)
        print(df.head(30))

        # 过滤数据，直到所有会话都具有足够的长度和频率

        '''
            测试--取消过滤
        '''
        df = self.filter_until_all_long_and_freq(df)
        print(df.head(30))

        # 如果目录不存在，则创建一个用于存储处理后的数据集的目录
        if not os.path.exists(f'E:/MyCode/PycharmCode/HG-GNN/data_processor/dataset/lastfm'):
            os.mkdir(f'E:/MyCode/PycharmCode/HG-GNN/data_processor/dataset/lastfm')

        # 将处理后的DataFrame保存为CSV文件
        df.to_csv(f'E:/MyCode/PycharmCode/HG-GNN/data_processor/dataset/lastfm/' + 'data.csv', sep=',', header=None,
                  index=False)

       # if not os.path.exists(f'{saved_path}/lastfm'):
       #      os.mkdir(f'{saved_path}/lastfm')
       #  df.to_csv(f'{saved_path}/lastfm/'+'data.txt',sep=',',header=None,index=False)


        # df_train, df_test = train_test_split(df, test_split=0.2)
        # save_dataset(dataset_dir, df_train, df_test)

    def _get_user_profile(self):
        '''
        获取用户配置文件。
        '''

        data_home = './raw_data/lastfm-1K'
        saved_path = './dataset'
        # csv_file = data_home + '/userid-timestamp-artid-artname-traid-traname.tsv'
        csv_file = 'E:/MyCode/PycharmCode/HG-GNN/data_processor/raw_data/lastfm-1K/userid-timestamp-artid-artname-traid-traname.tsv'
        # profile_file = data_home + '/userid-profile.tsv'
        profile_file = 'E:/MyCode/PycharmCode/HG-GNN/data_processor/raw_data/lastfm-1K/userid-profile.tsv'
        print(f'从 {csv_file} 中读取原始数据')

        # 读取CSV文件并创建DataFrame对象，包含LastFM数据。
        # DataFrame对象是pandas库中的一种数据结构，用于存储和操作二维数据。类似于电子表格或数据库表格，由行和列组成。
        df = pd.read_csv(
            csv_file,
            sep='\t',
            header=None,
            names=['userId', 'timestamp', 'itemId', 'artname', 'traid', 'traname'],
            usecols=['userId', 'timestamp', 'itemId'],
            parse_dates=['timestamp'],
            infer_datetime_format=True,
        )

        # 读取用户配置文件，并创建用户配置文件的DataFrame对象。
        user_df = pd.read_csv(
            profile_file,
            sep='\t'
            # names=['userId', 'timestamp', 'itemId', 'artname', 'traid', 'traname'],
            # usecols=['userId', 'timestamp', 'itemId'],
            # parse_dates=['timestamp'],
            # infer_datetime_format=True,
        )

        print('获取用户配置文件')
        # 删除DataFrame中含有缺失值（NaN）的行，确保数据的完整性。
        df = df.dropna()

        # 提取唯一的用户ID列表。
        uids = df['userId'].unique().tolist()

        # 将用户配置文件中的数据保存为CSV文件，仅保留包含在唯一用户ID列表中的用户数据。
        user_df[user_df['#id'].isin(uids)].to_csv(saved_path + '/lastfm/user_profile.csv', index=False)

        print('保存到 user_profile.csv 文件')

if __name__=='__main__':
    conf={
        'config': 'basic.ini'
    }

    config = configparser.ConfigParser()
    config.read("../basic.ini")
    config = config['default']
    # print(config.get("default", "dataset.name"))
    print(config['dataset.name'])
    lp=LastFM_Process(config)
    # 把数据转成dataformater格式   行列并排序分类  格式'userId', 'timestamp', 'itemId','sessionId'  生成 lastfm/data.txt
    lp._read_raw_data()

    # 读取上一步生成的data.txt
    lp._split_data()