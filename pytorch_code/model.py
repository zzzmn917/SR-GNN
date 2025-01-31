#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

# 得到物品的嵌入
import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F

# GNN 做的工作就是学习 item embedding ,对于矩阵 A 和 items 的 embedding结果，他返回一个 hidden 结果
class GNN(Module):
    # 初始化
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step  # 信息传递的层数， GNN 的迭代次数
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2  # 输入的维度大小，等于隐藏状态的两倍，因为考虑了入度和出度两个方向的信息
        self.gate_size = 3 * hidden_size  # 门控单元的维度大小，为 3 倍的隐藏状态大小，因为使用了三个门（重置门、更新门、新信息门）
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))  # 通过线性变换，用于计算输入到门控单元的权重
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))  # 通过线性变换，用于计算隐藏状态到门控单元的权重
        self.b_ih = Parameter(torch.Tensor(self.gate_size))   # 输入到门控单元的偏置项
        self.b_hh = Parameter(torch.Tensor(self.gate_size))   # 隐藏状态到门控单元的偏置项
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))  # 入度矩阵的偏置项
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))  # 出度矩阵的偏置项

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        # 公式1；第一个是入度矩阵，第二个是出度矩阵
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        # 拼接两个矩阵
        inputs = torch.cat([input_in, input_out], 2)
        # 不了解gi,gh
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        # 公式3 ? 有点对不上
        resetgate = torch.sigmoid(i_r + h_r)
        # 公式2 ？ 对不上
        inputgate = torch.sigmoid(i_i + h_i)
        # 公式4
        newgate = torch.tanh(i_n + resetgate * h_n)
        # 公式5 ？？？没看懂？？？
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class SessionGraph(Module):
    def __init__(self, opt, n_node):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize  # 定义隐藏层大小
        self.n_node = n_node
        self.batch_size = opt.batchSize  # 批处理大小
        self.nonhybrid = opt.nonhybrid
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)  # 嵌入层
        self.gnn = GNN(self.hidden_size, step=opt.step)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)   # 优化器； Lr 学习率
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()
    
    # 用于初始化模型参数，采用均匀分布初始化权重
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    # 公式8
    def compute_scores(self, hidden, mask):
        # 用于获取每个序列的最后一个有效项目的隐藏状态
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        # 对 ht 进行线性变换
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        # 对整个序列的隐藏状态 hidden 进行线性变换
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
        # 得到注意力权重
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        if not self.nonhybrid: # # 如果模型是混合模型
            a = self.linear_transform(torch.cat([a, ht], 1))
        # b 是通过索引获取嵌入层中项目的权重矩阵
        b = self.embedding.weight[1:]  # n_nodes x latent_size
        # 最终的推荐分数，a 乘以 b 的转置
        scores = torch.matmul(a, b.transpose(1, 0))
        return scores
    
    # 传递items=inputs ，A 两个变量给sessiongraph，调用这个forward
    # items使用 embedding 得到 hidden 变量，然后传递给GNN模型，调用GNN中的 forward
    def forward(self, inputs, A):
        hidden = self.embedding(inputs)
        hidden = self.gnn(A, hidden)
        return hidden


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable

# 前向传播函数
def forward(model, i, data):
    # 获取数据i的详细信息
    alias_inputs, A, items, mask, targets = data.get_slice(i)
    # 将获取到的数据转换为 PyTorch Tensor，并将其移到 GPU 上（如果可用）
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    A = trans_to_cuda(torch.Tensor(A).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    # 调用模型的前向传播函数，传入项目序列 items 和邻接矩阵 A，得到隐藏状态 hidden
    hidden = model(items, A)
    # 获取每个序列在隐藏状态中的表示，获取 item 的 embedding
    get = lambda i: hidden[i][alias_inputs[i]]
    # 生成 session 的 embedding 表示
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    return targets, model.compute_scores(seq_hidden, mask)


def train_test(model, train_data, test_data):
    model.scheduler.step()
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0   # 初始化总的损失为0
    # 生成batchs
    slices = train_data.generate_batch(model.batch_size)
    # 对于每个batch，遍历每个batch的slice
    for i, j in zip(slices, np.arange(len(slices))):
        # 首先梯度清0
        model.optimizer.zero_grad()   
        # 使用自己封装的forword进行前向传播得到预测的目标和分数
        targets, scores = forward(model, i, train_data)  
        targets = trans_to_cuda(torch.Tensor(targets).long())
        # 计算误差
        loss = model.loss_function(scores, targets - 1)
        # 反向传播求梯度
        loss.backward()  
        # 更新模型的参数
        model.optimizer.step()
        # 统计loss
        total_loss += loss
        # 训练时每隔一定步数输出一次损失
        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
    print('\tLoss:\t%.3f' % total_loss)
    
    # 模型评估，预测
    print('start predicting: ', datetime.datetime.now())
    # 将模型切换到评估模式-针对测试集
    model.eval()
    # 两个列表，用于记录推荐的命中率和平均倒数排名
    hit, mrr = [], [] 
    # 使用 test_data 生成测试集的 batch
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        # 使用前向传播得到预测的目标和分数
        targets, scores = forward(model, i, test_data)
        # 取分数最高的前20个推荐
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            # 判断真实目标是否在前20个推荐中，记录命中情况
            hit.append(np.isin(target - 1, score))
            # 计算平均倒数排名（MRR）
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
    # 对这两个指标进行的后续操作是：将其乘100转为百分比
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    return hit, mrr
