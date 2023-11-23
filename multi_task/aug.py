# -*- coding: utf-8 -*-
import os

import json

import numpy
import random

import networkx as nx
import os.path as osp
import shutil

import torch
from torch_geometric.data import InMemoryDataset, download_url, extract_zip, Dataset, Data
from torch_geometric.io import read_tu_data
from scipy.linalg import fractional_matrix_power, inv
from itertools import repeat, product
import numpy as np
from tu import read_tu_data_new
import os
import sys

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-4])
sys.path.append(root_path)


class HealthDataSet(InMemoryDataset):
    def __init__(self, root, name, cl_num_g=None, cl_num_n=None, transform=None, pre_transform=None,
                 pre_filter=None, use_node_attr=False, use_edge_attr=False,
                 cleaned=False, file_root=None):
        self.name = name
        self.cleaned = cleaned
        self.edge_weight = None
        self.cl_num_g = cl_num_g
        self.cl_num_n = cl_num_n
        self.file_root = file_root
        super(HealthDataSet, self).__init__(root, transform, pre_transform,
                                            pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        if self.data.x is not None and not use_node_attr:
            num_node_attributes = self.num_node_attributes
            # 方法1：只用了node_label来作为node_attributes
            # self.data.x = self.data.x[:, num_node_attributes:]
            # 方法2：采用随机正态分布的node_attribute+node_label来进行表示学习
            self.data.x = torch.cat((torch.normal(0.5, 0.1, (self.data.x.shape[0], num_node_attributes)),
                                     self.data.x[:, num_node_attributes:]), 1)
        if self.data.edge_attr is not None and not use_edge_attr:
            num_edge_attributes = self.num_edge_attributes
            self.data.edge_attr = self.data.edge_attr[:, num_edge_attributes:]

    @property
    def raw_dir(self):
        name = 'raw{}'.format('_cleaned' if self.cleaned else '')
        if self.file_root is not None:
            return osp.join(self.file_root, name)
        else:
            return osp.join(self.root, self.name, name)

    @property
    def processed_dir(self):
        name = 'processed{}'.format('_cleaned' if self.cleaned else '')
        if self.file_root is not None:
            return osp.join(self.file_root, name)
        else:
            return osp.join(self.root, self.name, name)

    @property
    def num_node_labels(self):
        if self.data.x is None:
            return 0
        for i in range(self.data.x.size(1)):
            x = self.data.x[:, i:]
            if ((x == 0) | (x == 1)).all() and (x.sum(dim=1) == 1).all():
                return self.data.x.size(1) - i
        return 0

    @property
    def num_node_attributes(self):
        if self.data.x is None:
            return 0
        return self.data.x.size(1) - self.num_node_labels

    @property
    def num_edge_labels(self):
        if self.data.edge_attr is None:
            return 0
        for i in range(self.data.edge_attr.size(1)):
            if self.data.edge_attr[:, i:].sum() == self.data.edge_attr.size(0):
                return self.data.edge_attr.size(1) - i
        return 0

    @property
    def num_edge_attributes(self):
        if self.data.edge_attr is None:
            return 0
        return self.data.edge_attr.size(1) - self.num_edge_labels

    @property  # 它返回一个包含没有处理的数据的名字的list。 该函数返回的文件名需要在raw_dir文件夹下找到才可以跳过下载过程
    def raw_file_names(self):
        # 为啥只读取A和graph_indicator
        # names = ['A', 'graph_indicator', 'graph_labels', 'node_labels','node_attributes']
        names = ['A', 'graph_indicator', 'graph_labels', 'node_attributes', 'node_labels', 'node_types']
        return ['{}_{}.txt'.format(self.name, name) for name in names]

    @property  # 该函数放回的文件名需要在processed_dir中找到才可以跳过处理过程。
    def processed_file_names(self):
        return 'data.pt'

    # 这个函数下载数据到你正在工作的目录中，你可以在self.raw_dir中指定。如果你不需要下载数据，你可以在这函数中简单的写一个 pass
    def download(self):
        pass

    #  这是Dataset中最重要的函数。你需要整合你的数据成一个包含data的list。然后调用 self.collate()去计算将用DataLodadr的片段
    # 处理原始数据并保存在processed_dir
    def process(self):
        self.data, self.slices = read_tu_data_new(self.raw_dir, self.name, self.cl_num_n)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))

    def get_num_feature(self):
        data = self.data.__class__()

        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[0]

        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key,
                                        item)] = slice(slices[0],
                                                       slices[0 + 1])
            else:
                # s = slice(slices[idx], slices[idx + 1])
                s = s  # 此处需要重写，cpu情况下的处理逻辑
            data[key] = item[s]
        _, num_feature = data.x.size()

        return num_feature

    def get(self, idx):
        data = self.data.__class__()

        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[idx]

        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key,
                                        item)] = slice(slices[idx],
                                                       slices[idx + 1])
            else:
                s = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]

        node_num = data.edge_index.max()
        sl = torch.tensor([[n, n] for n in range(node_num)]).t()
        data.edge_index = torch.cat((data.edge_index, sl), dim=1)

        return data


def drop_nodes(data):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    drop_num = int(node_num / 10)

    idx_drop = np.random.choice(node_num, drop_num, replace=False)
    idx_nondrop = [n for n in range(node_num) if not n in idx_drop]
    idx_dict = {idx_nondrop[n]: n for n in list(range(node_num - drop_num))}

    # data.x = data.x[idx_nondrop]
    edge_index = data.edge_index

    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    edge_index = adj.nonzero().t()

    data.edge_index = edge_index

    # edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)]
    # edge_index = [[edge_index[0, n], edge_index[1, n]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)] + [[n, n] for n in idx_nondrop]
    # data.edge_index = torch.tensor(edge_index).transpose_(0, 1)

    return data


def permute_edges(data):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    permute_num = int(edge_num / 10)

    edge_index = data.edge_index.transpose(0, 1)  # .numpy()

    idx_add = np.random.choice(node_num, (permute_num, 2))
    # idx_add = [[idx_add[0, n], idx_add[1, n]] for n in range(permute_num) if not (idx_add[0, n], idx_add[1, n]) in edge_index]

    # edge_index = np.concatenate((np.array([edge_index[n] for n in range(edge_num) if not n in np.random.choice(edge_num, permute_num, replace=False)]), idx_add), axis=0)
    # edge_index = np.concatenate((edge_index[np.random.choice(edge_num, edge_num-permute_num, replace=False)], idx_add), axis=0)
    edge_index = edge_index[np.random.choice(edge_num, edge_num - permute_num, replace=False)]
    # edge_index = [edge_index[n] for n in range(edge_num) if not n in np.random.choice(edge_num, permute_num, replace=False)] + idx_add
    # data.edge_index = torch.tensor(edge_index).transpose_(0, 1)
    data.edge_index = edge_index.transpose_(0, 1)

    return data


def subgraph(data):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    sub_num = int(node_num * 0.6)

    edge_index = data.edge_index

    idx_sub = [np.random.randint(node_num, size=1)[0]]
    idx_neigh = set([n for n in edge_index[1][edge_index[0] == idx_sub[0]]])

    count = 0
    while len(idx_sub) <= sub_num:
        count = count + 1
        if count > node_num:
            break
        if len(idx_neigh) == 0:
            break
        # sample_node = np.random.choice(list(idx_neigh))
        sample_node = np.random.choice(list(np.array(torch.tensor(list(idx_neigh), device='cpu'))))
        if sample_node in idx_sub:
            continue
        idx_sub.append(sample_node)
        idx_neigh.union(set([n for n in edge_index[1][edge_index[0] == idx_sub[-1]]]))

    idx_drop = [n for n in range(node_num) if not n in idx_sub]
    idx_nondrop = idx_sub
    idx_dict = {idx_nondrop[n]: n for n in list(range(len(idx_nondrop)))}

    # data.x = data.x[idx_nondrop]
    edge_index = data.edge_index

    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    edge_index = adj.nonzero().t()

    data.edge_index = edge_index

    # edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)]
    # edge_index = [[edge_index[0, n], edge_index[1, n]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)] + [[n, n] for n in idx_nondrop]
    # data.edge_index = torch.tensor(edge_index).transpose_(0, 1)

    return data


def mask_nodes(data):
    node_num, feat_dim = data.x.size()
    mask_num = int(node_num / 10)

    idx_mask = np.random.choice(node_num, mask_num, replace=False)
    data.x[idx_mask] = torch.tensor(np.random.normal(loc=0.5, scale=0.5, size=(mask_num, feat_dim)),
                                    dtype=torch.float32).cuda()

    return data


def substitute_nodes_bak(data, node_content_dict, word_to_ix, ix_to_word, vector):
    node_num, feat_dim = data.x.size()
    mask_num = int(node_num / 4)

    idx_mask = np.random.choice(node_num, mask_num, replace=False)
    if len(idx_mask) > 0:
        data.x[idx_mask] = torch.as_tensor(
            find_nearby_node(data.x[idx_mask], node_content_dict, word_to_ix, ix_to_word, vector),
            dtype=torch.float32).cuda()
    return data


def to_str(var):
    return str(list(np.reshape(np.asarray(var), (1, np.size(var)))[0]))[1:-1]


def substitute_nodes(data, node_content_dict, word_to_ix, ix_to_word, vector, random_parent_node_list):
    node_num, feat_dim = data.x.size()
    mask_num = int(node_num / 4)

    idx_mask = np.random.choice(node_num, mask_num, replace=False)
    if len(idx_mask) > 0:
        data.x[idx_mask] = torch.as_tensor(
            find_nearby_node(data.x[idx_mask], node_content_dict, word_to_ix, ix_to_word, vector,
                             random_parent_node_list),
            dtype=torch.float32).cuda()
    return data


# 根据字典的值value获得该值对应的key
def get_dict_key(dic, value):
    for key in dic.keys():
        if (dic[key] == value).all():  #
            return key
    return False


def find_nearby_node(node, node_content_dict, word_to_ix, ix_to_word, vector, random_parent_node_list):
    # 根据原有节点的嵌入表达转换为原始字符串表达
    # NumArray = str2num(node_emb, comment='#')
    # 如何保证每个aug的数据和原有的对应
    # 6,16,96，根据具体的数据和dim设置需要调整，后续改为参数设置，当前glove训练的dim=16，最大节点长度为6，node的lable长度为2
    node_emb = node[:, 0:96].reshape(node.__len__(), 6, 16)  # 这里可能为多个节点
    node_emb_label = node[:, 96:]

    # 得到每个node的verb对应的ix
    substitute_node_emb_list = []
    not_in_num = 0
    for i in range(len(node_emb)):
        to_be_find = to_str(node_emb[i].view(1,96).cpu().tolist()[0])
        if to_str(node_emb[i].cpu().tolist()) in random_parent_node_list.keys():
            #sub_node_emb = random_parent_node_list[to_str(node_emb[i].cpu().tolist())]
            sub_node_emb = random_parent_node_list[to_be_find]
        else:  # if not find, keep un-change
            sub_node_emb=to_be_find
        sub_node_emb = sub_node_emb.replace("\n", "").split(",")
        sub_node_emb = list(map(float, sub_node_emb))
        substitute_node_emb_list.append(sub_node_emb)

    substitute_node_emb = numpy.array(substitute_node_emb_list)
    substitute_node_emb = torch.FloatTensor(substitute_node_emb)
    substitute_node_result = torch.cat((substitute_node_emb.view((substitute_node_emb.shape[0],
                                                                  substitute_node_emb.shape[1])),
                                        node_emb_label.cpu()), 1)

    return substitute_node_result


def find_nearby_node_bak(node, node_content_dict, word_to_ix, ix_to_word, vector):
    # 根据原有节点的嵌入表达转换为原始字符串表达
    # NumArray = str2num(node_emb, comment='#')
    # 如何保证每个aug的数据和原有的对应
    # 6,16,96，根据具体的数据和dim设置需要调整，后续改为参数设置，当前glove训练的dim=16，最大节点长度为6，node的lable长度为2
    node_emb = node[:, 0:96].reshape(node.__len__(), 6, 16)  # 这里可能为多个节点
    node_emb_label = node[:, 96:]

    # 得到每个node的verb对应的ix
    org_attr_ix_list = []
    for i in range(len(node_emb)):
        bb = []
        for j in range(len(node_emb[i])):
            if (node_emb[i][j].max() != 0):
                aa = get_dict_key(vector, numpy.array(node_emb[i][j].tolist()).astype(numpy.float32))
                if aa:
                    bb.append(aa)
        org_attr_ix_list.append(bb)

    # 得到每个node的ix_to_word
    org_attr_word_list = []
    for i in range(len(org_attr_ix_list)):
        org_str = ''
        for attr_ix in org_attr_ix_list[i]:
            # print(ix_to_word[attr])
            if org_str == '':
                org_str = ix_to_word[attr_ix]
            else:
                org_str = org_str + '|' + ix_to_word[attr_ix]
        org_attr_word_list.append(org_str.replace('\n', ''))
    # print(org_attr_word_list)

    substitute_node_emb_list = []
    for str_list in org_attr_word_list:
        str_list = str_list.replace('\n', '').split('|')
        len_str = len(str_list)

        # 选择具有相同父节点的属性集合作为候选的list
        node_candidate = []
        for key, _ in node_content_dict.items():
            if node_content_dict[key][2] - len_str >= 0:
                flag = 1
                for i in range(len_str - 1):
                    if node_content_dict[key][3 + i] != str_list[i]:
                        flag = flag * 0
                if flag == 1:
                    node_candidate.append(key)

        select_id = random.randrange(0, len(node_candidate), 1)
        select_node_str = node_candidate[select_id]

        # 得到候选的embedding
        select_node_list = select_node_str.split('|')
        substitute_node_emb = []
        for att in select_node_list:
            substitute_node_emb.append(vector[word_to_ix[str(att)]])

        max_len = 6
        max_dim = 16
        if len(substitute_node_emb) < max_len:
            for i in range(max_len - len(substitute_node_emb)):
                substitute_node_emb.append(np.zeros(max_dim))
        substitute_node_emb_list.append(substitute_node_emb)

    substitute_node_emb = numpy.array(substitute_node_emb_list)
    substitute_node_emb = torch.FloatTensor(substitute_node_emb)
    substitute_node_result = torch.cat((substitute_node_emb.view((substitute_node_emb.shape[0],
                                                                  substitute_node_emb.shape[1] *
                                                                  substitute_node_emb.shape[2])),
                                        node_emb_label.cpu()), 1)

    return substitute_node_result


def diffgraph(data):
    # diff 是对边的权重进行了更新 #
    node_num, _ = data.x.size()
    edge_index = data.edge_index  # .numpy()

    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    edge_weight = torch.FloatTensor(compute_ppr(adj.tolist()))
    data.edge_weight = edge_weight[edge_index[0], edge_index[1]]  # 无边的的地方没有权重
    return data


def compute_ppr(a, alpha=0.2, self_loop=False):
    if self_loop:
        a = a + np.eye(a.shape[0])  # A^ = A + I_n
    d = np.diag(np.sum(a, 1))  # D^ = Sigma A^_ii
    dinv = fractional_matrix_power(d, -0.5)  # D^(-1/2)
    at = np.matmul(np.matmul(dinv, a), dinv)  # A~ = D^(-1/2) x A^ x D^(-1/2)
    # return alpha * inv((np.eye(a.shape[0]) - (1 - alpha) * at))   # a(I_n-(1-a)A~)^-1
    return alpha * inv((np.eye(len(a)) - (1 - alpha) * at))


def compute_heat(a, t=5, self_loop=True):
    if self_loop:
        a = a + np.eye(a.shape[0])
    d = np.diag(np.sum(a, 1))
    return np.exp(t * (np.matmul(a, inv(d)) - 1))
