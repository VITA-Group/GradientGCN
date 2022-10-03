import os
import random
from warnings import warn

import numpy as np
import torch
import yaml
from texttable import Texttable
import numpy as np
import scipy
import scipy.linalg as LA
import torch_geometric.transforms as T
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigs
from torch_geometric.data.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import get_laplacian
from torch_geometric.utils.convert import from_networkx

def tonp(tsr):
    if isinstance(tsr, np.ndarray):
        return tsr
    elif isinstance(tsr, np.matrix):
        return np.array(tsr)
    elif isinstance(tsr, scipy.sparse.csc.csc_matrix):
        return np.array(tsr.todense())

    assert isinstance(tsr, torch.Tensor)
    tsr = tsr.cpu()
    assert isinstance(tsr, torch.Tensor)

    try:
        arr = tsr.numpy()
    except TypeError:
        arr = tsr.detach().to_dense().numpy()
    except:
        arr = tsr.detach().numpy()

    assert isinstance(arr, np.ndarray)
    return arr

def get_laplacian_mat(edge_index,  num_node, normalization='sym'):  # todo: change back
    """ return a laplacian (torch.sparse.tensor)"""
    edge_index, edge_weight = get_laplacian(edge_index, normalization=normalization)  # see https://bit.ly/3c70FJK for format
    return torch.sparse.FloatTensor(edge_index, edge_weight, torch.Size([num_node, num_node]))

def energy(v1, L1):
    """ compute the energy
        v1: n * d
        L1 : n * n
        return tr(v.T * L * v)
    """

    L1 = tonp(L1)
    assert v1.shape[0] == L1.shape[0] == L1.shape[1]
    E = np.dot(np.dot(v1.T, L1), v1)
    E = np.diag(E)
    return np.sum(E)

def print_args(args):
    _dict = vars(args)
    _key = sorted(_dict.items(), key=lambda x: x[0])
    t = Texttable()
    t.add_row(["Parameter", "Value"])
    for k, _ in _key:
        t.add_row([k, _dict[k]])
    print(t.draw())


def set_seed(args):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.cuda and not torch.cuda.is_available():  # cuda is not available
        args.cuda = False
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_num)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)


def AcontainsB(A, listB):
    # A: string; listB: list of strings
    for s in listB:
        if s in A: return True
    return False


def yaml_parser(model):
    filename = os.path.join('options/configs', f'{model}.yml')
    if os.path.exists(filename):
        with open(filename, 'r') as yaml_f:
            configs = yaml.load(yaml_f, Loader=yaml.FullLoader)
        return configs
    else:
        warn(f'configs of {model} not found, use the default setting instead')
        return {}


def overwrite_with_yaml(args, model, dataset):
    configs = yaml_parser(model)
    if dataset not in configs.keys():
        warn(f'{model} have no specific settings on {dataset}. Use the default setting instead.')
        return args
    for k, v in configs[dataset].items():
        if k in args.__dict__:
            args.__dict__[k] = v
        else:
            warn(f"Ignored unknown parameter {k} in yaml.")
    return args
