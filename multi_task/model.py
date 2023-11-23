import os
import time

import numpy
from torch import nn

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from aug import drop_nodes, permute_edges, subgraph, mask_nodes, diffgraph, substitute_nodes

from torch_geometric.nn import global_add_pool
from torch_geometric.nn import GINConv, GCNConv, GATConv, SAGEConv, TransformerConv
from torch_geometric.nn.models.mlp import MLP
from torch_geometric.data.batch import Batch
from copy import deepcopy
import re, ast, json

import numpy as np


class Encoder(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers, encoder_model):
        super(Encoder, self).__init__()

        # num_features = dataset.num_features
        # dim = 32
        self.num_gc_layers = num_gc_layers
        self.encoder_model = encoder_model

        # self.nns = []
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for i in range(num_gc_layers):

            if self.encoder_model == 'GINConv':
                if i:
                    nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
                else:
                    nn = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
                conv = GINConv(nn)
            elif self.encoder_model == 'GATConv':
                if i:
                    conv = GATConv(in_channels=dim, out_channels=dim, heads=3, concat=False)
                else:
                    conv = GATConv(in_channels=num_features, out_channels=dim, heads=3, concat=False)
            elif self.encoder_model == 'GCNConv':
                if i:
                    conv = GCNConv(in_channels=dim, out_channels=dim)
                else:
                    conv = GCNConv(in_channels=num_features, out_channels=dim)
            elif self.encoder_model == 'SAGEConv':
                if i:
                    conv = SAGEConv(in_channels=dim, out_channels=dim)
                else:
                    conv = SAGEConv(in_channels=num_features, out_channels=dim)
            elif self.encoder_model == 'TransformerConv':
                if i:
                    conv = TransformerConv(in_channels=dim, out_channels=dim)
                else:
                    conv = TransformerConv(in_channels=num_features, out_channels=dim)
            elif self.encoder_model == 'MLP':
                if i:
                    conv = MLP(channel_list=[dim, dim])
                else:
                    conv = MLP(channel_list=[num_features, dim])

            bn = torch.nn.BatchNorm1d(dim)

            self.convs.append(conv)
            self.bns.append(bn)

    def forward(self, x, edge_index, batch, edge_weight=None):
        if x is None:
            x = torch.ones((batch.shape[0], 1))
            x = torch.FloatTensor(x).cuda()

        xs = []
        if edge_weight is not None:
            edge_weight = edge_weight.cuda()

        for i in range(self.num_gc_layers):
            if self.encoder_model == 'GCNConv':
                x = F.relu(self.convs[i](x, edge_index, edge_weight=edge_weight))
            elif self.encoder_model == 'MLP':
                x = F.relu(self.convs[i](x))
            else:
                x = F.relu(self.convs[i](x, edge_index))
            x = self.bns[i](x)
            xs.append(x)

        return torch.cat(xs, 1)


class Expert(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        # self.log_soft = nn.LogSoftmax(1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        # out = self.log_soft(out)
        return out


class Tower(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Tower, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        # self.softmax = nn.Softmax(dim=1)
        # self.sigmoid = nn.Sigmoid()
        # self.LogSoftmax = nn.LogSoftmax

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        # out = self.softmax(out)
        # out = self.sigmoid(out)
        # out = self.LogSoftmax(out, -1)

        return out


class healthCRL(nn.Module):
    def __init__(self, hidden_dim, num_gc_layers, encoder_model, dataset_num_features, cl_num_g, cl_num_n,
                 data_set, is_contrastive, tasks_g_index, num_experts, is_task_node_cl, aug1, aug2, alpha=0.5, beta=1.,
                 gamma=.1):
        super(healthCRL, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.data_set = data_set
        self.is_contrastive = is_contrastive
        self.aug1 = aug1
        self.aug2 = aug2
        self.cl_num_g = cl_num_g
        self.cl_num_n = cl_num_n
        self.tasks_num_n = 1
        self.tasks_g_index = tasks_g_index
        self.tasks_num_g = len(tasks_g_index)
        self.is_task_node_cl = is_task_node_cl
        self.hidden_dim = hidden_dim
        self.embedding_dim = hidden_dim * num_gc_layers
        self.random_parent_node_list = None

        if len(self.tasks_g_index) == 1:
            self.proj_head = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                           nn.Linear(self.embedding_dim, self.embedding_dim))
            self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers, encoder_model)
            self.lin_class_g = nn.Linear(self.embedding_dim, cl_num_g)

            if self.is_contrastive == 1:
                self.proj_head_n = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim),
                                                 nn.ReLU(inplace=True),
                                                 nn.Linear(self.embedding_dim, self.embedding_dim))

        # if len(self.tasks_g_index) == 4:
        if len(self.tasks_g_index) >1:
            self.towers_hidden = self.embedding_dim
            self.experts_num = num_experts
            self.experts_hidden = self.embedding_dim
            self.experts_out = self.embedding_dim
            self.lin_class_g = nn.Linear(self.embedding_dim, cl_num_g)
            self.proj_head = nn.ModuleList(
                [nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                               nn.Linear(self.embedding_dim, self.embedding_dim)) for i in range(self.experts_num)])
            self.proj_head_n = nn.ModuleList(
                [nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                               nn.Linear(self.embedding_dim, self.embedding_dim)) for i in range(self.experts_num)])

            self.experts = nn.ModuleList(
                [Encoder(dataset_num_features, hidden_dim, num_gc_layers, encoder_model) for i in
                 range(self.experts_num)])  # 目前有4个graph级别的任务，所以是4个expert
            self.w_gates = nn.ParameterList(
                [nn.Parameter(torch.randn(dataset_num_features, self.experts_num), requires_grad=True) for i in
                 range(self.tasks_num_g)])
            self.towers = nn.ModuleList(
                [Tower(self.experts_out, cl_num_g, self.towers_hidden) for i in range(self.tasks_num_g)])
            self.towers_n = nn.ModuleList([Tower(self.experts_out, cl_num_n, self.towers_hidden) for i in range(1)])

            # self.tasks_g_i = torch.nn.ModuleList()
            # for i in range(len(self.tasks_g_index)):
            #     self.tasks_g_i.append(nn.Linear(self.embedding_dim, cl_num_g))

        if self.is_contrastive == 1:
            self.init_node_content_dict()
            self.init_ix_to_word()
            self.init_word_to_ix()
            self.init_get_vector()
            if self.aug1 == 'substitute_nodes' or self.aug2 == 'substitute_nodes':
                self.init_parent_embedding()

        self.init_emb()
        self.softmax = nn.Softmax(dim=1)

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, data, aug1, aug2):  # x, edge_index, batch, num_graphs
        # batch_size = data.num_graphs
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        if x is None:
            x = torch.ones(batch.shape[0]).cuda()

        # g, n = self.encoder(x, edge_index, batch)

        pred_g = None
        pred_n_out = None
        aug_data_list = []
        if self.is_contrastive == 1:
            x_aug1, edge_index_aug1, batch_aug1, edge_weight1 = self.get_augmented_data(data, aug1)
            x_aug2, edge_index_aug2, batch_aug2, edge_weight2 = self.get_augmented_data(data, aug2)
            # if len(self.tasks_g_index) == 4:
            if len(self.tasks_g_index) >1:
                # for i in self.tasks_g_index:
                for i in range(self.experts_num):
                    n1 = self.experts[i](x_aug1, edge_index_aug1, batch_aug1, edge_weight1)
                    n2 = self.experts[i](x_aug2, edge_index_aug2, batch_aug2, edge_weight2)
                    g1 = self.read_out(n1.split(self.hidden_dim, 1), batch_aug1)
                    g2 = self.read_out(n2.split(self.hidden_dim, 1), batch_aug2)
                    g1 = self.proj_head[i](g1)
                    g2 = self.proj_head[i](g2)
                    n1 = self.proj_head_n[i](n1)
                    n2 = self.proj_head_n[i](n2)
                    aug_data_list.append([g1, g2, n1, n2, batch_aug1, batch_aug2])
            elif len(self.tasks_g_index) == 1:
                n1 = self.encoder(x_aug1, edge_index_aug1, batch_aug1, edge_weight1)
                n2 = self.encoder(x_aug2, edge_index_aug2, batch_aug2, edge_weight2)
                g1 = self.read_out(n1.split(self.hidden_dim, 1), batch_aug1)
                g2 = self.read_out(n2.split(self.hidden_dim, 1), batch_aug2)
                g1 = self.proj_head(g1)
                g2 = self.proj_head(g2)
                n1 = self.proj_head_n(n1)
                n2 = self.proj_head_n(n2)
                aug_data_list.append([g1, g2, n1, n2, batch_aug1, batch_aug2])

        # if len(self.tasks_g_index) == 4:
        if len(self.tasks_g_index) >1:
            expers_o = [e(x, edge_index, batch) for e in self.experts]
            expers_o_tensor = torch.stack(expers_o)
            gates_o = [self.softmax(x @ wg) for wg in self.w_gates]
            # multiply the output of the experts with the corresponding gates output
            towers_input = [g.t().unsqueeze(2).expand(-1, -1, self.experts_out) * expers_o_tensor for g in gates_o]
            towers_input = [torch.sum(ti, dim=0) for ti in towers_input]
            # get the final output from the towers for graph level tasks
            towers_input_g = [self.read_out(ti.split(64, 1), batch) for ti in towers_input]
            pred_g = [t(ti) for t, ti in zip(self.towers, towers_input_g)]
            for i in range(len(pred_g)):
                pred_g[i] = F.log_softmax(pred_g[i], dim=-1)
            # get the final output from the towers for node level tasks
            towers_input_n = towers_input
            pred_n = [t(ti) for t, ti in zip(self.towers_n, towers_input_n)]
            pred_n[0] = F.softmax(pred_n[0], dim=1)
            pred_n_out = pred_n[0]

        # only for graph node level task
        if len(self.tasks_g_index) == 1 and self.is_task_node_cl == 0:
            n = self.encoder(x, edge_index, batch)
            g = self.read_out(n.split(64, 1), batch)
            pred_g = [[], [], [], []]
            g = self.proj_head(g)
            y_pred = self.lin_class_g(g)
            pred_g[self.tasks_g_index[0]] = F.log_softmax(y_pred, dim=-1)

        return aug_data_list, pred_n_out, pred_g

    def read_out(self, input, batch):
        # tmp = input[0].split(64, 1)
        tmp = [global_add_pool(x, batch) for x in input]
        x = torch.cat(tmp, 1)
        return x

    def get_augmented_data(self, data, aug):
        if aug == 'None':
            data_aug = data
        else:
            data_aug = get_aug_data_set(data, aug, self.node_content_dict, self.word_to_ix, self.ix_to_word,
                                        self.vector, self.random_parent_node_list)
        x_aug = data_aug.x
        edge_index_aug = data_aug.edge_index
        batch_aug = data_aug.batch
        edge_weight = None
        if aug == 'diff':
            edge_weight = data_aug.edge_weight

        return x_aug, edge_index_aug, batch_aug, edge_weight

    def get_embeddings(self, data):
        # x = torch.FloatTensor(data.x).cuda()
        # edge_index = torch.LongTensor(data.edge_index).cuda()
        # batch = torch.LongTensor(data.batch).cuda()

        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        # n = self.encoder(x, edge_index, batch)

        pred_g = None
        pred_n_out = None
        embedding_g = None
        if len(self.tasks_g_index) > 1:
            expers_o = [e(x, edge_index, batch) for e in self.experts]
            expers_o_tensor = torch.stack(expers_o)
            gates_o = [self.softmax(x @ wg) for wg in self.w_gates]
            # multiply the output of the experts with the corresponding gates output
            towers_input = [g.t().unsqueeze(2).expand(-1, -1, self.experts_out) * expers_o_tensor for g in gates_o]
            towers_input = [torch.sum(ti, dim=0) for ti in towers_input]
            # get the final output from the towers for graph level tasks
            towers_input_g = [self.read_out(ti.split(64, 1), batch) for ti in towers_input]
            embedding_g = [self.read_out(ti.split(64, 1), batch) for ti in expers_o]

            pred_g = [t(ti) for t, ti in zip(self.towers, towers_input_g)]

            for i in range(len(pred_g)):
                pred_g[i] = F.log_softmax(pred_g[i], dim=-1)
            # get the final output from the towers for node level tasks
            towers_input_n = towers_input
            pred_n = [t(ti) for t, ti in zip(self.towers_n, towers_input_n)]
            pred_n[0] = F.softmax(pred_n[0], dim=1)
            pred_n_out = pred_n[0]

        # only for graph single task
        if len(self.tasks_g_index) == 1 and self.is_task_node_cl == 0:
            n = self.encoder(x, edge_index, batch)
            g = self.read_out(n.split(64, 1), batch)
            pred_g = [[], [], [], []]
            g = self.proj_head(g)
            embedding_g = g
            y_pred = self.lin_class_g(g)
            pred_g[self.tasks_g_index[0]] = F.log_softmax(y_pred, dim=-1)

        return pred_n_out, pred_g, embedding_g

    def loss_cal(self, x, x_aug):
        T = 0.8
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()

        return loss

    def init_node_content_dict(self):
        raw_data_path = '/home/project/GraphCLHealth/processed_data/' + self.data_set + '/full/raw/' + self.data_set + '_node_seqs.txt'
        i = 0
        inff = open(raw_data_path, 'r')
        node_content_dict = {}

        for line in inff.readlines():
            node_seq = line.split('|@|')
            node_id = node_seq[0]
            node_content = node_seq[1]
            node_type = node_seq[2]
            node_str2int = node_seq[3]
            graph_id = node_seq[4]

            tmp = node_content.split('|')
            if node_content not in node_content_dict:
                node_content_dict[node_content] = []
                node_content_dict[node_content].append(node_type)
                node_content_dict[node_content].append(int(1))
                node_content_dict[node_content].append(int(len(tmp)))
                for nn in tmp:
                    node_content_dict[node_content].append(nn)
            else:
                node_content_dict[node_content][1] += 1
            i += 1
        inff.close
        self.node_content_dict = node_content_dict

    def init_word_to_ix(self):
        raw_data_path = '/home/project/GraphCLHealth/data_pre/glove/export/' + self.data_set + '/word_to_ix.txt'
        inff = open(raw_data_path, 'r')
        word_to_ix = {}
        for line in inff.readlines():
            line_seq = line.split('|@|')
            word_to_ix[line_seq[0]] = int(line_seq[1].replace('\n', ''))
        inff.close
        self.word_to_ix = word_to_ix

    def init_ix_to_word(self):
        raw_data_path = '/home/project/GraphCLHealth/data_pre/glove/export/' + self.data_set + '/ix_to_word.txt'

        inff = open(raw_data_path, 'r')
        ix_to_word = {}
        for line in inff.readlines():
            line_seq = line.split('|@|')
            ix_to_word[int(line_seq[0])] = line_seq[1]
        inff.close
        self.ix_to_word = ix_to_word

    def init_get_vector(self):
        raw_data_path = '/home/project/GraphCLHealth/data_pre/glove/export/' + self.data_set + '/Vector.txt'
        inff = open(raw_data_path, 'r')
        vector = {}
        for line in inff.readlines():
            a = '[' + line.replace(' ', ',') + ']'
            a = json.loads(a)
            a = numpy.array(a)
            id = int(a[0:1])
            # vector[id]=a[1:]
            # print((a[1:].tolist())) # ndarray to list
            vector[id] = (a[1:]).astype(numpy.float32)
            # if((a[1:]==node_emb[0]).all()):
            #    print('sss')
        inff.close
        self.vector = vector

    def init_parent_embedding(self):
        raw_data_path = '/home/project/GraphCLHealth/processed_data/' + self.data_set + '/full/raw/' + self.data_set + '_random_parent_node_seqs.txt'
        inff = open(raw_data_path, 'r')
        random_parent_node_list = {}
        for line in inff.readlines():
            node_embeding = line.split('|@|')
            #random_parent_node_list.append([node_embeding[0],node_embeding[1]])
            if node_embeding[0] not in random_parent_node_list.keys():
                random_parent_node_list[node_embeding[0]]=node_embeding[1]

        inff.close
        self.random_parent_node_list = random_parent_node_list


def init_node_list_dict(data_set):
    raw_data_path = '/home/project/GraphCLHealth/processed_data/' + data_set + '/full/raw/' + data_set + '_node_seqs.txt'
    i = 0
    inff = open(raw_data_path, 'r')
    node_list = {}

    for line in inff.readlines():
        node_seq = line.split('|@|')
        node_id = node_seq[0]
        node_content = node_seq[1]
        node_type = node_seq[2]
        node_str2int = node_seq[3]
        graph_id = node_seq[4]

        node_list[node_id] = []
        node_list[node_id].append(node_content)
        node_list[node_id].append(node_type)
        node_list[node_id].append(node_str2int)
        node_list[node_id].append(graph_id)
    inff.close
    return node_list


# gat augmentation by batch
def get_aug_data_set(data_org, aug_type, node_content_dict, word_to_ix, ix_to_word, vector, random_parent_node_list):
    data_list = []
    data_org_list = Batch.to_data_list(data_org)

    for data in data_org_list:
        data = get_aug_data(data, aug_type, node_content_dict, word_to_ix, ix_to_word, vector, random_parent_node_list)
        data_list.append(data.cuda())
    data_aug = Batch.from_data_list(data_list)

    if aug_type == 'dnodes' or aug_type == 'subgraph' or aug_type == 'random2' or aug_type == 'random3' or aug_type == 'random4':
        # node_num_aug, _ = data_aug.x.size()
        edge_idx = data_aug.edge_index.cpu().numpy()
        _, edge_num = edge_idx.shape
        node_num, _ = data_org.x.size()
        idx_not_missing = [n for n in range(node_num) if (n in edge_idx[0] or n in edge_idx[1])]

        node_num_aug = len(idx_not_missing)
        data_aug.x = data_aug.x[idx_not_missing]

        data_aug.batch = data_org.batch[idx_not_missing]
        idx_dict = {idx_not_missing[n]: n for n in range(node_num_aug)}
        edge_idx = [[idx_dict[edge_idx[0, n]], idx_dict[edge_idx[1, n]]] for n in range(edge_num) if
                    not edge_idx[0, n] == edge_idx[1, n]]
        data_aug.edge_index = torch.tensor(edge_idx).transpose_(0, 1).cuda()

    return data_aug


# focal_loss func, L = -α(1-yi)**γ *ce_loss(xi,yi)
class Focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=3, size_average=True):
        super(Focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)

        self.gamma = gamma

    def forward(self, preds, labels):
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1, preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_softmax = F.softmax(preds, dim=1)
        preds_logsoft = torch.log(preds_softmax)

        # focal_loss func, Loss = -α(1-yi)**γ *ce_loss(xi,yi)
        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        self.alpha = self.alpha.gather(0, labels.view(-1))
        # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)

        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


# gat augmentation by data
def get_aug_data(data_aug, aug_type, node_content_dict, word_to_ix, ix_to_word, vector, random_parent_node_list):
    if aug_type == 'dnodes':
        data_aug = drop_nodes(data_aug)
    elif aug_type == 'pedges':
        data_aug = permute_edges(data_aug)
    elif aug_type == 'subgraph':
        data_aug = subgraph(data_aug)
    elif aug_type == 'mask_nodes':
        data_aug = mask_nodes(data_aug)
    elif aug_type == 'diff':
        data_aug = diffgraph(data_aug)
    elif aug_type == 'substitute_nodes':
        data_aug = substitute_nodes(data_aug, node_content_dict, word_to_ix, ix_to_word, vector, random_parent_node_list)
    elif aug_type == 'none':
        data_aug = deepcopy(data_aug)
        data_aug.x = torch.ones((data_aug.edge_index.max() + 1, 1))
    elif aug_type == 'random2':
        n = np.random.randint(2)
        if n == 0:
            data_aug = drop_nodes(deepcopy(data_aug))
        elif n == 1:
            data_aug = subgraph(deepcopy(data_aug))
        else:
            print('sample error')
            assert False
    elif aug_type == 'random3':
        n = np.random.randint(3)
        if n == 0:
            data_aug = drop_nodes(deepcopy(data_aug))
        elif n == 1:
            data_aug = permute_edges(deepcopy(data_aug))
        elif n == 2:
            data_aug = subgraph(deepcopy(data_aug))
        else:
            print('sample error')
            assert False
    elif aug_type == 'random4':
        n = np.random.randint(4)
        if n == 0:
            data_aug = drop_nodes(deepcopy(data_aug))
        elif n == 1:
            data_aug = permute_edges(deepcopy(data_aug))
        elif n == 2:
            data_aug = subgraph(deepcopy(data_aug))
        elif n == 3:
            data_aug = mask_nodes(deepcopy(data_aug))
        else:
            print('sample error')
            assert False
    elif aug_type == 'random6':
        n = np.random.randint(4)
        if n == 0:
            data_aug = drop_nodes(deepcopy(data_aug))
        elif n == 1:
            data_aug = permute_edges(deepcopy(data_aug))
        elif n == 2:
            data_aug = subgraph(deepcopy(data_aug))
        elif n == 3:
            data_aug = mask_nodes(deepcopy(data_aug))
        elif n == 4:
            data_aug = substitute_nodes(data_aug, node_content_dict, word_to_ix, ix_to_word, vector, random_parent_node_list)
        elif n == 5:
            data_aug = diffgraph(data_aug)
        else:
            print('sample error')
            assert False
    else:
        print('augmentation error')
        assert False

    return data_aug
