import torch
import numpy as np
import random
import torch.nn.functional as F

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

# Borrowed from https://github.com/fanyun-sun/InfoGraph
def get_positive_expectation(p_samples, measure, average=True):
    """Computes the positive part of a divergence / difference.
    Args:
        p_samples: Positive samples.
        measure: Measure to compute for.
        average: Average the result over samples.
    Returns:
        torch.Tensor
    """
    log_2 = np.log(2.)

    if measure == 'GAN':
        Ep = - F.softplus(-p_samples)
    elif measure == 'JSD':
        Ep = log_2 - F.softplus(- p_samples)
    elif measure == 'X2':
        Ep = p_samples ** 2
    elif measure == 'KL':
        Ep = p_samples + 1.
    elif measure == 'RKL':
        Ep = -torch.exp(-p_samples)
    elif measure == 'DV':
        Ep = p_samples
    elif measure == 'H2':
        Ep = 1. - torch.exp(-p_samples)
    elif measure == 'W1':
        Ep = p_samples

    if average:
        return Ep.mean()
    else:
        return Ep


# Borrowed from https://github.com/fanyun-sun/InfoGraph
def get_negative_expectation(q_samples, measure, average=True):
    """Computes the negative part of a divergence / difference.
    Args:
        q_samples: Negative samples.
        measure: Measure to compute for.
        average: Average the result over samples.
    Returns:
        torch.Tensor
    """
    log_2 = np.log(2.)

    if measure == 'GAN':
        Eq = F.softplus(-q_samples) + q_samples
    elif measure == 'JSD':
        Eq = F.softplus(-q_samples) + q_samples - log_2
    elif measure == 'X2':
        Eq = -0.5 * ((torch.sqrt(q_samples ** 2) + 1.) ** 2)
    elif measure == 'KL':
        Eq = torch.exp(q_samples)
    elif measure == 'RKL':
        Eq = q_samples - 1.
    elif measure == 'H2':
        Eq = torch.exp(q_samples) - 1.
    elif measure == 'W1':
        Eq = q_samples

    if average:
        return Eq.mean()
    else:
        return Eq


# Borrowed from https://github.com/fanyun-sun/InfoGraph
def local_global_loss_(l_enc, g_enc, batch, measure):
    '''
    Args:
        l: Local feature map.
        g: Global features.
        measure: Type of f-divergence. For use with mode `fd`
        mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.
    Returns:
        torch.Tensor: Loss.
    '''
    num_graphs = g_enc.shape[0]
    num_nodes = l_enc.shape[0]
    max_nodes = num_nodes // num_graphs

    pos_mask = torch.zeros((num_nodes, num_graphs)).cuda()
    neg_mask = torch.ones((num_nodes, num_graphs)).cuda()
    msk = torch.ones((num_nodes, num_graphs)).cuda()
    index_ = [[i for i in range(batch.__len__())], batch.tolist()]
    pos_mask[index_] = 1.
    neg_mask[index_] = 0.
    # for nodeidx, graphidx in enumerate(batch): # 这个循环比较慢
    #     pos_mask[nodeidx][graphidx] = 1.
    #     neg_mask[nodeidx][graphidx] = 0.

    res = torch.mm(l_enc, g_enc.t())

    E_pos = get_positive_expectation(res * pos_mask, measure, average=False).sum()
    E_pos = E_pos / num_nodes
    E_neg = get_negative_expectation(res * neg_mask, measure, average=False).sum()
    E_neg = E_neg / (num_nodes * (num_graphs - 1))
    return E_neg - E_pos




def global_global_loss_(g1_enc, g2_enc, measure):
    '''
    Args:
        l: Local feature map.
        g: Global features.
        measure: Type of f-divergence. For use with mode `fd`
        mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.
    Returns:
        torch.Tensor: Loss.
    '''
    num_graphs = g1_enc.shape[0]

    pos_mask = torch.zeros((num_graphs, num_graphs)).cuda()
    neg_mask = torch.ones((num_graphs, num_graphs)).cuda()
    for graphidx in range(num_graphs):
        pos_mask[graphidx][graphidx] = 1.
        neg_mask[graphidx][graphidx] = 0.

    res = torch.mm(g1_enc, g2_enc.t())

    E_pos = get_positive_expectation(res * pos_mask, measure, average=False).sum()
    E_pos = E_pos / num_graphs
    E_neg = get_negative_expectation(res * neg_mask, measure, average=False).sum()
    E_neg = E_neg / (num_graphs * (num_graphs - 1))
    return E_neg - E_pos

def select_train_valid_test(dataset, random_seed=1234):
    train_size = int(0.8 * len(dataset))
    val_size =   int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size,val_size, test_size])
    return train_dataset, val_dataset, test_dataset


def str2num(LineString, comment='#'):
    from io import StringIO as StringIO
    import re, numpy

    NumArray = numpy.empty([0], numpy.int16)
    NumStr = LineString.strip()
    # ~ ignore comment string
    for cmt in comment:
        CmtRe = cmt + '.*$'
        NumStr = re.sub(CmtRe, " ", NumStr.strip(), count=0, flags=re.IGNORECASE)

    # ~ delete all non-number characters,replaced by blankspace.
    NumStr = re.sub('[^0-9.e+-]', " ", NumStr, count=0, flags=re.IGNORECASE)

    # ~ Remove incorrect combining-characters for double type.
    NumStr = re.sub('[.e+-](?=\s)', " ", NumStr.strip(), count=0, flags=re.IGNORECASE)
    NumStr = re.sub('[.e+-](?=\s)', " ", NumStr.strip(), count=0, flags=re.IGNORECASE)
    NumStr = re.sub('[e+-]$', " ", NumStr.strip(), count=0, flags=re.IGNORECASE)
    NumStr = re.sub('[e+-]$', " ", NumStr.strip(), count=0, flags=re.IGNORECASE)

    if len(NumStr.strip()) > 0:
        StrIOds = StringIO(NumStr.strip())
        NumArray = numpy.genfromtxt(StrIOds)

    return NumArray