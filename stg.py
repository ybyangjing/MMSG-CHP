'''
gow pre-process:
implement of merged gowalla data to spatial-temporal graph
'''
import copy
import matplotlib.pyplot as plt

import random
import pickle

import geohash2
import numpy as np
import pandas as pd
import networkx as nx

import torch
from geopy.distance import geodesic
from tqdm import tqdm
from units import get_distance_hav
from model.time_encoder import PositionalEncoding, TimeEmbedding,timestamp_to_features
import warnings

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth', 100)
d_model = 16
time_embedding = TimeEmbedding(d_model)

cates = ['Coffee Shop'] #'Coffee Shop', 'Sandwich Shop'

gowalla_graph_num = np.zeros(700)
P = 0.05  # for calculating threshold0.01/0.05/0.1/0.5
ghash_code_index = 'bcfguvyz89destwx2367kmqr0145hjnp'
geo_dict = dict(zip(ghash_code_index, range(32)))
friend_path = '/home/zy/mambaGAT/gowalla_friendship.csv'

# Save path
sub_net_path = '/media/zy/UBUNTU 20_0/0.05cos_sub_net_'
n_feat_path = '/media/zy/UBUNTU 20_0/0.05cos_n_feat_'
e_feat_path = '/media/zy/UBUNTU 20_0/0.05cos_e_feat_'
label_path = '/media/zy/UBUNTU 20_0/0.05cos_label_'


def Get_dis(geo1, geo2):
    lat1, lon1 = geohash2.decode(geo1)
    lat2, lon2 = geohash2.decode(geo2)
    loc1 = (lat1, lon1)
    loc2 = (lat2, lon2)
    d = geodesic(loc1, loc2).kilometers
    return d

def aug_topology(sim_mx, adjacency_matrix, weight_matrix=None, percent=0.2, new_geo=None, time=None):
    """Generate the data augmentation from topology (graph structure) perspective
        for weighted directed graph without self-loop.
    :param sim_mx: tensor, symmetric similarity, [v,v]
    :param adjacency_matrix: tensor, adjacency matrix without self-loop, [v,v]
    :param weight_matrix: tensor, weight matrix for edges, [v,v]
    :param percent: float, percentage of edges to modify (half dropped, half added)
    :param new_geo: dict, a dictionary mapping node to its geographic location
    :param time: dict, a dictionary mapping node to its time
    :return aug_adjacency_matrix: tensor, augmented adjacency matrix with weights, [v,v]
    """
    if adjacency_matrix.sum() == 0:
        # 如果图是空的，直接返回原始的邻接矩阵和权重矩阵（或者返回零矩阵）
        return adjacency_matrix, weight_matrix if weight_matrix is not None else torch.zeros_like(adjacency_matrix)

    # 获取所有边的索引
    index_list = adjacency_matrix.nonzero(as_tuple=False)
    edge_num = index_list.shape[0] // 2  # 计算无向图的边数

    edge_mask = (adjacency_matrix > 0).tril(diagonal=-1)
    aug_adjacency_matrix = copy.deepcopy(adjacency_matrix)

    if weight_matrix is not None:
        aug_weight_matrix = copy.deepcopy(weight_matrix)
    else:
        aug_weight_matrix = torch.zeros_like(adjacency_matrix)  # 如果没有提供权重矩阵，初始化为全零矩阵

    # 计算需要删除和添加的边数
    num_edges_to_modify = int(edge_num * percent / 2)

    # 确保删除边和添加边的数量相同
    drop_prob = (1. - torch.softmax(sim_mx[edge_mask], dim=0)).numpy()
    drop_prob /= drop_prob.sum()

    add_prob = torch.softmax(sim_mx[torch.ones(sim_mx.size(), dtype=bool).tril(diagonal=-1)], dim=0).numpy()
    add_prob /= add_prob.sum()

    # 删除边
    if num_edges_to_modify > 0:
        drop_list = cdf_sampling(drop_prob, num_edges_to_modify)
        drop_index = index_list[drop_list]

        zeros = torch.zeros_like(aug_adjacency_matrix[0, 0])
        aug_adjacency_matrix[drop_index[:, 0], drop_index[:, 1]] = zeros
        aug_adjacency_matrix[drop_index[:, 1], drop_index[:, 0]] = zeros

        aug_weight_matrix[drop_index[:, 0], drop_index[:, 1]] = zeros
        aug_weight_matrix[drop_index[:, 1], drop_index[:, 0]] = zeros

        # 添加新的边
        node_num = adjacency_matrix.shape[0]
        x, y = np.meshgrid(range(node_num), range(node_num), indexing='ij')
        mask = y < x
        x, y = x[mask], y[mask]

        add_list = cdf_sampling(add_prob, num_edges_to_modify)

        for i in add_list:
            node1, node2 = x[i], y[i]

            # 确保只在原有节点之间添加边
            if node1.item() >= node_num or node2.item() >= node_num:
                continue  # 跳过不在原有节点集合中的节点

            d = Get_dis(new_geo[node1.item()], new_geo[node2.item()])  # 计算朋友之间的距离

            if time[node1.item()] > time[node2.item()]:
                aug_adjacency_matrix[node2, node1] = 1
                aug_weight_matrix[node2, node1] = d
            else:
                aug_adjacency_matrix[node1, node2] = 1
                aug_weight_matrix[node1, node2] = d

    # 最后检查并修正节点数
    # 确保最终图的节点数与原始图相同
    if aug_adjacency_matrix.shape[0] != adjacency_matrix.shape[0]:
        print("Warning: Augmented graph has more nodes than the original graph.")
        aug_adjacency_matrix = aug_adjacency_matrix[:adjacency_matrix.shape[0], :adjacency_matrix.shape[0]]
        aug_weight_matrix = aug_weight_matrix[:adjacency_matrix.shape[0], :adjacency_matrix.shape[0]]

    return aug_adjacency_matrix, aug_weight_matrix


def cdf_sampling(probs, num_samples):
    if len(probs) == 0:
        # 如果输入的概率数组为空，返回空的采样结果
        return np.array([], dtype=int)
    cdf = np.cumsum(probs)
    cdf /= cdf[-1]
    random_values = np.random.rand(num_samples)
    return np.searchsorted(cdf, random_values)


# def process_graph(nx_g, new_geo, time):
#     if nx_g is None or len(nx_g.edges) == 0:
#         # 如果图是空的或没有边，直接返回原始的图
#         return nx_g
#     else:
#         adjacency_matrix = nx.to_numpy_array(nx_g)
#         weight_matrix = nx.to_numpy_array(nx_g, weight='weight')
#         adjacency_matrix = torch.tensor(adjacency_matrix)
#         weight_matrix = torch.tensor(weight_matrix)
#
#     similarity_matrix = torch.rand((adjacency_matrix.shape[0], adjacency_matrix.shape[1]))
#     similarity_matrix = (similarity_matrix + similarity_matrix.t()) / 2
#
#     # 增强拓扑结构
#     augmented_adjacency_matrix, augmented_weight_matrix = aug_topology(
#         similarity_matrix, adjacency_matrix, weight_matrix=weight_matrix, percent=0.2, new_geo=new_geo, time=time
#     )
#
#     # 创建有向图
#     augmented_nx_g = nx.DiGraph()
#
#     # 确保增强后的图与原始图的节点保持一致
#     nodes_mapping = {i: n for i, n in enumerate(nx_g.nodes())}
#     augmented_nx_g.add_nodes_from(nx_g.nodes(data=True))
#
#     # 添加边及其权重
#     for i, j in zip(*np.where(augmented_weight_matrix.numpy() > 0)):
#         node_i = nodes_mapping[i]
#         node_j = nodes_mapping[j]
#         augmented_nx_g.add_edge(node_i, node_j, weight=augmented_weight_matrix[i, j].item())
#
#     return augmented_nx_g
def process_graph(nx_g, node_features, new_geo, time):
    if len(nx_g.edges) == 0:
        # 如果图是空的，直接返回原始的图
        return nx_g
    else:
        adjacency_matrix = nx.to_numpy_array(nx_g)
        weight_matrix = nx.to_numpy_array(nx_g, weight='weight')
        adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float32)
        weight_matrix = torch.tensor(weight_matrix, dtype=torch.float32)

    # 将one_period_feature转换为torch.Tensor
    node_features = torch.tensor(node_features, dtype=torch.float32)

    # 计算余弦相似度矩阵
    similarity_matrix = torch.nn.functional.cosine_similarity(
        node_features.unsqueeze(1), node_features.unsqueeze(0), dim=2
    )

    # 增强拓扑结构
    augmented_adjacency_matrix, augmented_weight_matrix = aug_topology(
        similarity_matrix, adjacency_matrix, weight_matrix=weight_matrix, percent=0.2, new_geo=new_geo, time=time
    )

    # 创建有向图
    augmented_nx_g = nx.DiGraph()

    # 确保增强后的图与原始图的节点保持一致
    nodes_mapping = {i: n for i, n in enumerate(nx_g.nodes())}
    augmented_nx_g.add_nodes_from(nx_g.nodes(data=True))

    # 添加边及其权重
    for i, j in zip(*np.where(augmented_weight_matrix.numpy() > 0)):
        node_i = nodes_mapping[i]
        node_j = nodes_mapping[j]
        augmented_nx_g.add_edge(node_i, node_j, weight=augmented_weight_matrix[i, j].item())

    return augmented_nx_g
def net_complete():  # Read the data and call other functional functions for processing
    data_path = '/home/zy/mambaGAT/merged_data.csv'  # the data sorted by category after merging data
    merged_data = pd.read_csv(data_path)
    merged_data = merged_data[['userid', 'lng', 'lat', 't', 'geo']]

    with open("/home/zy/mambaGAT/Gow_cate_dict.pkl", 'rb') as f:
        spot_id_dict = pickle.load(f)  # 类别数据在merge_data中第一次出现的位置

    cate_keys = list(spot_id_dict.keys())  # 地点类别，就是Coffee Shop这些
    row_num = list(
        spot_id_dict.values())  # 得到了对应Coffee Shop的键值为[90, 8816858]第90类,从8816858开始为第90类的记录；merged_data里对应类别记录的起始序号
    for (index, key) in zip(range(len(cate_keys)), cate_keys):  # len(cate_keys))为630 index,key就是cate里面的类别和对应的序号
        if key not in cates:
            continue
        cate_sub_net_path = sub_net_path + str(index) + '.pkl'
        cate_n_feat_path = n_feat_path + str(index) + '.pkl'
        cate_e_feat_path = e_feat_path + str(index) + '.pkl'
        cate_label_path = label_path + str(index) + '.pkl'
        cate_save_path = [cate_sub_net_path, cate_n_feat_path, cate_e_feat_path, cate_label_path]
        if index == len(cate_keys) - 1:  # 如果是最后一类
            raw_data = merged_data.iloc[row_num[index][1]:]  # 就把key这个类别的所有记录取出来放到raw_data
        else:  # 如果不是最后一类
            raw_data = merged_data.iloc[row_num[index][1]:row_num[index + 1][1]]  # 就把这个类别的所有记录到下个类别开始前的记录取出来
            # raw_data就是某一个类别在merged_data的所有记录
        random.seed(30)
        chosen = np.sort(random.sample(range(len(raw_data)),
                                       min(300000, len(raw_data))))  # 从raw_data中随机选取最多500000个不重复的元素，将这些元素按索引排序。
        # random.sample(range(len(raw_data)), min(500000, len(raw_data)))的意思是从range(len(raw_data))范围内选取min(500000, len(raw_data))个不重复元素的索引
        raw_data = raw_data.iloc[chosen]  # 选取raw_data的包含chosen的行
        pro_data = raw_data.drop_duplicates(keep='first')  # 删除重复行，保留第一个出现行。
        pro_data.sort_values(by="geo", inplace=True)  # 对pro_data按照geo排序，不创建新副本，就在原来的dataframe改
        pro_data = pro_data.reset_index().iloc[:, 1:]  # 重置索引并移除第一列
        time_attr = pro_data['t']  # 获取pro_data['t']的数据
        t_min = time_attr.min()
        t_max = time_attr.max()
        time_index = []
        for j in np.linspace(t_min, t_max, 11):  # 从t_min到t_max结束，生成11个等间距数值
            time_index.append(np.float64(j))
        time_index.remove(time_index[0])
        print('read done:{}'.format(index))
        print('num of data lines:{}'.format(len(pro_data)))
        pro_data = np.array(pro_data.values).astype(np.str_)
        one_cate = Static_graph(pro_data)
        sub_net_sampling(one_cate, time_index, cate_save_path, index)  # one_cate=G,time,geo index是类别的序号


def Static_graph(pro_data):
    """
    :param pro_data: All Check records under the same location category (gowalla )
    :return: G(networkx graph),time,gps
    """
    friend_ship = pd.read_csv(friend_path)
    friend_ship = np.array(friend_ship.values)  # 转为numpy数组
    g_tmp = nx.Graph()
    g_tmp.add_edges_from(friend_ship)

    user_id = list(pro_data[:, 0].astype(np.float64))
    gps = list(pro_data[:, [2, 1]].astype(np.float64))
    time_ = list(pro_data[:, -2].astype(np.float64))
    geo = list(pro_data[:, -1])

    new_g_tmp = g_tmp.subgraph(user_id)  # 从g_tmp中把userid表所包含的所有id作为一张图提取出来,对应边页提取出来了
    G = nx.DiGraph()
    G.add_nodes_from(range(len(user_id)))
    # transformed user name into an ID, record the corresponding check-in node
    ud2rc = {}  # user_id to record(node)
    for (index, u) in zip(range(len(user_id)), user_id):
        ud2rc.setdefault(u, []).append(index)  # 如果u没在字典里，就把u和空列表加入字典作为对应的键值，并把index加入列表，当第二次出现u时就直接把index加入列表,记住同一user_id的位置
        # 归类相同id
    for u1 in tqdm(new_g_tmp.nodes()):  # 字典中存放的是每个userid在userid列表中出现的位置，也就这个userid的地理位置在gps中出现的位置
        ner = list(new_g_tmp.neighbors(u1))  # 获取u1节点的“邻接节点”列表
        # Randomly select 35% of friends
        # (https://www.businessinsider.com/35-percent-of-friends-see-your-facebook-posts-2013-8)
        for u2 in random.sample(ner, int(0.35 * len(ner))):  # 从ner中获取35%的u1的邻接节点
            for node1 in ud2rc[u1]:  # 所以node1就是当前处理的userid在gps表中出现的位置
                for node2 in ud2rc[u2]:  # u2是u1的邻居，node2就是u2在gps表中的位置
                    d = get_distance_hav(gps[node1], gps[node2])  # 计算朋友之间的距离
                    if time_[node1] > time_[node2]:  # 所以time[node1]就是当前处理的userid在time的位置
                        G.add_edge(node2, node1, weight=d)
                    else:
                        G.add_edge(node1, node2, weight=d)

    print('Static Graph, node nums:{}, edge nums:{}'.format(G.number_of_nodes(), G.number_of_edges()))

    return G, time_, geo


def sub_net_sampling(args, time_index, save_path, cate_index):
    # Converts a category's data into a spatial-temporal graph
    net = args[0]
    nodes_all = list(net.nodes)
    time_ = np.array(args[1])
    geo = np.array(args[2])
    node_num = len(nodes_all)
    sub_net = []
    node_feature = []
    edge_feature = []
    label = []
    random.seed(23)
    chosen = np.sort(random.sample(net.nodes(), min(20000, int(node_num))))  # 从net.nodes()中随机选最少的节点，并排序
    dup_dict = {}
    net_index = 0

    for i in tqdm(chosen):  # 关联周边区域的边

        if geo[i] in dup_dict.keys():  # geo[i]代表的是列表中第i个坐标
            continue
        nodes = list(range(max(0, i - 1000), min(i + 1000, node_num)))  # 考虑节点i的node范围内的索引,因为是按地理位置排序的
        for j in reversed(nodes):  # 去除不满足地理位置条件的点
            if geo[j][:5] != geo[i][:5]:  # 比较j和i的地理位置字符编码的前五位是否不相同
                nodes.remove(j)  # 移除和i地理位置不同的节点

        dup_dict.setdefault(geo[i], []).append(net_index)  # 生成了每一个地理位置一共有多少节点的字典 ,dup_dict就是区域的数目，每个区域最后有一个label
        net_index += 1  # 在chosen的节点中包含的区域数目

        sequence_net = []
        sequence_feature = []
        sequence_edge = []
        sequence_label = []
        for j in range(len(time_index)):  # get rid of the points that don't satisfy the time condition
            sub_nodes = nodes.copy()
            for t in reversed(sub_nodes):
                if time_[t] > time_index[j] or (time_[t] < time_index[j - 1] if j != 0 else False):
                    sub_nodes.remove(t)

            one_period_label = np.zeros(32)
            one_period_feature = np.zeros((len(sub_nodes), 48))

            for index, t in enumerate(sub_nodes):
                if j == len(time_index) - 1:
                    one_period_label[geo_dict[geo[t][-1]]] += 1
                one_period_feature[index, geo_dict[geo[t][-1]]] = 1
            if len(sub_nodes)!= 0:
                for index, t in enumerate(sub_nodes):
                    time_f = timestamp_to_features(time_[t]) #转换为年月日时分秒
                    time_f = torch.tensor([[
                        time_f['hour'],
                        time_f['minute'],
                        time_f['second'],
                        time_f['day'],
                        time_f['month'],
                        time_f['year'],
                        time_f['weekday']
                    ]])
                    t_embed = time_embedding(time_f)
                    t_embed_numpy = t_embed.detach().cpu().numpy()
                    t_embed_numpy_flat = t_embed_numpy.flatten()

                    for i in range(len(t_embed_numpy_flat)):
                        one_period_feature[index, 32 + i] = t_embed_numpy_flat[i]



            # ag=process_graph(net.subgraph(sub_nodes),geo,time_)
            ag = process_graph(net.subgraph(sub_nodes), one_period_feature, geo, time_)
            # one_period_net = net.subgraph(sub_nodes)
            mapping = dict(zip(ag, range(len(ag.nodes()))))
            ag = nx.relabel_nodes(ag, mapping)

            # plt.figure(figsize=(8, 6))
            # pos = nx.spring_layout(ag)  # 使用 spring 布局
            # nx.draw(ag, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=15)
            # plt.title("Relabeled Graph Visualization")
            # plt.show()
            # get d (edge feature)
            attr_tmp = list(ag.edges.values())
            attr_tmp = np.array([x['weight'] for x in attr_tmp])
            if len(attr_tmp) > 0:
                attr_min = attr_tmp.min()
                attr_max = attr_tmp.max()
                if attr_max != 0:
                    print(attr_max)
                attr_tmp = [(x - attr_min) / ((attr_max - attr_min) if (attr_max - attr_min) else 1) for x in attr_tmp]
            sequence_edge.append(attr_tmp)

            # get net and feature
            sequence_net.append(ag)
            sequence_feature.append(one_period_feature)

            # get label
            if j == len(time_index) - 1:
                # # Record the last for the tag
                thr = one_period_label.mean() + \
                      ((one_period_label.sum() - one_period_label.mean()) * P)
                # print("thr:{}".format(thr))
                thr = max(thr, 0.6)

                one_period_label = np.int64(one_period_label >= thr)
                sequence_label.append(one_period_label)

        sub_net.append(sequence_net)
        node_feature.append(sequence_feature)
        edge_feature.append(sequence_edge)
        label.append(sequence_label[0])
    print('dup dict len:', len(dup_dict.values()))

    if len(sub_net) != 0:
        print('st-graphs num:{}, node_feat_list:{}, '
              '\n edge_feat_list:{}, label_list:{}'.format(len(sub_net), len(node_feature), len(edge_feature),
                                                           len(label)))
        with open(save_path[0], 'wb') as f:
            pickle.dump(sub_net, f)
        with open(save_path[1], 'wb') as f:
            pickle.dump(node_feature, f)
        with open(save_path[2], 'wb') as f:
            pickle.dump(edge_feature, f)
        with open(save_path[3], 'wb') as f:
            pickle.dump(label, f)

        # dup=list(dup_dict.keys())
        # file_name1="dup.pkl"
        # with open(file_name1, 'wb') as f:
        #     pickle.dump(dup, f)


def main():
    net_complete()
    print('done')


if __name__ == '__main__':
    main()
