import os
import os.path as osp
import glob

import torch
import torch.nn.functional as F
import numpy as np
from torch_sparse import coalesce
from torch_geometric.io import read_txt_array
from torch_geometric.utils import remove_self_loops
from torch_geometric.data import Data

names = [
    'A', 'graph_indicator', 'node_labels', 'node_attributes','mimiciii_node_types'
    'edge_labels', 'edge_attributes', 'graph_labels', 'graph_attributes', 'graph_labels_expired',
    'graph_labels_readmission','graph_labels_los_3day','graph_labels_los_7day', 'graph_labels'
]


def read_tu_data_new(folder, prefix, node_classes_num):
    files = glob.glob(osp.join(folder, '{}_*.txt'.format(prefix)))
    names = [f.split(os.sep)[-1][len(prefix) + 1:-4] for f in files]

    edge_index = read_file(folder, prefix, 'A', torch.long).t() - 1
    batch = read_file(folder, prefix, 'graph_indicator', torch.long) - 1

    node_attributes = node_labels = node_types = None

    if 'node_attributes' in names:
        node_attributes = read_file(folder, prefix, 'node_attributes')
    if 'node_types' in names:
        node_types = read_file(folder, prefix, 'node_types', torch.long)
        if node_types.dim() == 1:
            node_types = node_types.unsqueeze(-1)
        node_types = node_types - node_types.min(dim=0)[0]
        node_types = node_types.unbind(dim=-1)
        node_types = [F.one_hot(x, num_classes=-1) for x in node_types]
        node_types = torch.cat(node_types, dim=-1).to(torch.float)
    x = cat([node_attributes, node_types])

    if 'node_labels' in names:
        node_labels = read_file(folder, prefix, 'node_labels', torch.long)
        _, node_labels = node_labels.unique(sorted=True, return_inverse=True)
        if node_labels.dim() == 1:
            node_labels = node_labels.unsqueeze(-1)
        node_labels = node_labels - node_labels.min(dim=0)[0]
        node_labels = node_labels.unbind(dim=-1)
        # node_labels = [F.one_hot(x, num_classes=-1) for x in node_labels]
        ## 此语句主要防止spit后标签缺失导致的one_hot不一致的情况
        node_labels = [F.one_hot(torch.arange(0, node_classes_num), num_classes=-1)[x] for x in node_labels]
        node_labels = torch.cat(node_labels, dim=-1).to(torch.float)

    edge_attributes, edge_labels = None, None
    if 'edge_attributes' in names:
        edge_attributes = read_file(folder, prefix, 'edge_attributes')
    if 'edge_labels' in names:
        edge_labels = read_file(folder, prefix, 'edge_labels', torch.long)
        if edge_labels.dim() == 1:
            edge_labels = edge_labels.unsqueeze(-1)
        edge_labels = edge_labels - edge_labels.min(dim=0)[0]
        edge_labels = edge_labels.unbind(dim=-1)
        edge_labels = [F.one_hot(e, num_classes=-1) for e in edge_labels]
        edge_labels = torch.cat(edge_labels, dim=-1).to(torch.float)
    edge_attr = cat([edge_attributes, edge_labels])

    y  = None
    if 'graph_labels' in names:  # Regression problem.
        y = read_file(folder, prefix, 'graph_labels')
    # Mortality LOS3 LOS7 Readmin

    # if task=='Mortality' and 'graph_labels_expired' in names:
    #     y = read_file(folder, prefix, 'graph_labels_expired', torch.long)
    # if task=='LOS3' and 'graph_labels_los_3day' in names:
    #     y = read_file(folder, prefix, 'graph_labels_los_3day', torch.long)
    # if task=='LOS7' and 'graph_labels_los_7day' in names:
    #     y = read_file(folder, prefix, 'graph_labels_los_7day', torch.long)
    # if task=='Readmin' and 'graph_labels_readmission' in names:
    #     y = read_file(folder, prefix, 'graph_labels_readmission', torch.long)
    _, y = y.unique(sorted=True, return_inverse=True)

    num_nodes = edge_index.max().item() + 1 if x is None else x.size(0)
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes,
                                     num_nodes)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, node_labels = node_labels, node_types = node_types)
    data, slices = split(data, batch)

    return data, slices


def read_file(folder, prefix, name, dtype=None):
    path = osp.join(folder, '{}_{}.txt'.format(prefix, name))
    return read_txt_array(path, sep=',', dtype=dtype)


def cat(seq):
    seq = [item for item in seq if item is not None]
    seq = [item.unsqueeze(-1) if item.dim() == 1 else item for item in seq]
    return torch.cat(seq, dim=-1) if len(seq) > 0 else None


def split(data, batch):
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])

    row, _ = data.edge_index
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])

    # Edge indices should start at zero for every graph.
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)

    slices = {'edge_index': edge_slice}
    if data.x is not None:
        slices['x'] = node_slice
    else:
        # Imitate `collate` functionality:
        data._num_nodes = torch.bincount(batch).tolist()
        data.num_nodes = batch.numel()
    if data.edge_attr is not None:
        slices['edge_attr'] = edge_slice
    if data.y is not None:
        if data.y.size(0) == batch.size(0):
            slices['y'] = node_slice
        else:
            slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)
    # if data.node_labels is not None:
    #     slices['node_labels'] = node_slice
    if data.node_labels is not None:
        slices['node_labels'] = node_slice
    if data.node_types is not None:
        slices['node_types'] = node_slice

    return data, slices
