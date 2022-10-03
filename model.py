from layers import *
from torch import nn

from torch_geometric.utils import get_laplacian
import scipy
import scipy.linalg as LA

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


class DeepGCNII(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nlayer, adj = None,  residual=0,
                 norm_mode='None', norm_scale=1, **kwargs):
        super(DeepGCNII, self).__init__()
        self.adj = adj
        assert nlayer >= 1 
        self.hidden_layers = nn.ModuleList([
            GraphConv(nfeat if i==0 else nhid, nhid, self.adj) 
            for i in range(nlayer-1)
        ])
        self.out_layer = GraphConv(nfeat if nlayer==1 else nhid , nclass, self.adj)
        self.fc = nn.Linear(nfeat, nclass)
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_rate = dropout
        self.relu = nn.ReLU(True)
        self.skip = residual
        self.L = None
        self.norm = PairNorm(norm_mode, norm_scale)
        self.skip_list = None

    
    def forward(self, x, adj, edge_index, isVal):
        if self.L == None:
            self.L = get_laplacian_mat(edge_index,  x.shape[0])
        x_init = None
        alpha = 1
        fopen = open("./plots/de_cora_10_init1.txt", "a")
        energy_str = ""
        for i, layer in enumerate(self.hidden_layers):
            x = self.dropout(x)
            x = layer(x, adj)
            if i == 0:
                x_init = x.clone()
            if i in self.skip_list:
                x = x + alpha * x_init
            e = energy(tonp(x.clone()), self.L)
            energy_str += str(e) + " "
            x = self.relu(x)
        x = self.dropout(x)
        x = self.out_layer(x, adj)
        fopen.write(energy_str + "\n")
        fopen.flush()
        return x, None
