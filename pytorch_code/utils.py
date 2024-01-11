#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import networkx as nx
import numpy as np

# 为训练集中的序列构建有向图，并且为其中的边赋予权重（见论文记录本内容）
def build_graph(train_data):
    graph = nx.DiGraph()
    # 构建有向图
    for seq in train_data:
        for i in range(len(seq) - 1):
            if graph.get_edge_data(seq[i], seq[i + 1]) is None:
                weight = 1
            else:
                weight = graph.get_edge_data(seq[i], seq[i + 1])['weight'] + 1
            graph.add_edge(seq[i], seq[i + 1], weight=weight)
    # 用于规范化入边的权重（指向同一节点的所有边权重之和为1）        
    for node in graph.nodes:
        sum = 0
        for j, i in graph.in_edges(node):
            sum += graph.get_edge_data(j, i)['weight']
        if sum != 0:
            for j, i in graph.in_edges(i):
                graph.add_edge(j, i, weight=graph.get_edge_data(j, i)['weight'] / sum)
    return graph

# 补全session长度
def data_masks(all_usr_pois, item_tail):  # all_usr_pois: 一个包含用户行为序列的列表；item_tail: 表示用于填充不足长度的虚拟项目
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens) # 用户行为序列的最大长度
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)] # 处理后的用户行为序列列表，每个序列的长度都被填充到最大长度
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens] # 表示用户行为序列的实际长度
    return us_pois, us_msks, len_max

# 用于将训练集划分为训练集和测试集
def split_validation(train_set, valid_portion): # valid_portion：表示测试集占总训练集的比例。
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)   #上面两行代码，用于随机打乱训练集的顺序
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]  # 划分后的测试集输入
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]  # 测试集的输出

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


class Data():
    # 初始化，对输入的数据进行预处理
    def __init__(self, data, shuffle=False, graph=None):
        inputs = data[0]
        inputs, mask, len_max = data_masks(inputs, [0])
        self.inputs = np.asarray(inputs)
        self.mask = np.asarray(mask)
        self.len_max = len_max
        self.targets = np.asarray(data[1])
        self.length = len(inputs)
        self.shuffle = shuffle
        self.graph = graph
    
    # 用于生成批次数据，可以指定批次大小
    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        return slices
    
    # 对输入的处理包括获取唯一项目、构建邻接矩阵等
    def get_slice(self, i):
        inputs, mask, targets = self.inputs[i], self.mask[i], self.targets[i]
        items, n_node, A, alias_inputs = [], [], [], []
        for u_input in inputs:  # np.unique(u_input) 来获取每个用户输入序列中的唯一项目
            n_node.append(len(np.unique(u_input))) 
        max_n_node = np.max(n_node)
        for u_input in inputs:
            node = np.unique(u_input)
            items.append(node.tolist() + (max_n_node - len(node)) * [0]) # 将当前用户输入序列中的项目列表转换为一个固定长度的列表（max_n_node），如果不足的话，用 0 补齐
            u_A = np.zeros((max_n_node, max_n_node)) # 创建一个零矩阵，用于存储邻接矩阵
            for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0:
                    break
                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                u_A[u][v] = 1 # 在邻接矩阵中标记节点 u 指向节点 v 的关系，即建立了两个项目之间的邻接关系
            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1  # 计算节点入度的权重总和，若是0则写为1
            u_A_in = np.divide(u_A, u_sum_in) # 计算输入节点的归一化权重矩阵，即将每个节点的入度按总和进行归一化
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1  # 同上
            u_A_out = np.divide(u_A.transpose(), u_sum_out)  # 归一化输出节点的权重矩阵
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A.append(u_A)  # A：归一化后的出入度邻接矩阵
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input]) # alias_input :session中物品对应于去重排序后的session的id
        return alias_inputs, A, items, mask, targets # items：排序去重补全后的序列； mask：填充后的session,item id换为1，填充位是0
