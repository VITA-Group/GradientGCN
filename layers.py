from statistics import stdev
import torch
import torch.nn as nn
from torch_geometric import seed_everything 
from torch_sparse import spmm # require the newest torch_sprase
import numpy as np 
import sys
from tqdm import tqdm
import random

class GraphConv(nn.Module):
    def __init__(self, in_features, out_features, adj = None, bias=True):
        super(GraphConv, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(1, out_features))
        self.adj = adj
       
        self.in_features = in_features
        self.out_features = out_features
        self.reset_parameters()
    
    def get_std(self, C):
        print(C)
        N = self.adj.shape[0]
        d_i = torch.sparse.sum(self.adj, dim = 1).to_dense().unsqueeze(1) + 1
        d_j = torch.sparse.sum(self.adj, dim = 0).to_dense().unsqueeze(0) + 1
        support = torch.sqrt(torch.sum(d_i * d_j))
        stdv = N/torch.sqrt(torch.sum(d_i * d_j) * 3)
        # stdv = torch.sqrt( (N * N) / (3 * C * torch.sum(d_i * d_j)))
        # stdv = (N * 1.732)/(C * support)
        return stdv

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        if self.adj != None:
            
            stdv = self.get_std(self.weight.shape[1])
            print("Using new initialization : stdv - {}".format(stdv))
            for i in range(0, self.weight.shape[0]):
                self.weight[i].data.uniform_(-stdv, stdv)
            if self.bias is not None:
                self.bias.data.uniform_(-stdv, stdv)
        else:
            print("Using normal initialization: stdv - {}".format(stdv))
            self.weight.data.uniform_(-stdv, stdv)
            if self.bias is not None:
                self.bias.data.uniform_(-stdv, stdv)
            
    def forward(self, input, adj):
        h = torch.mm(input, self.weight)
        output = torch.spmm(adj, h)
        if self.bias is not None:
            return output + self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + "({}->{})".format(
                    self.in_features, self.out_features)

class GraphAttConv(nn.Module):
    def __init__(self, in_features, out_features, heads, dropout, _adj):
        super(GraphAttConv, self).__init__()
        assert out_features % heads == 0
        out_perhead = out_features // heads

        self.graph_atts = nn.ModuleList([GraphAttConvOneHead(
               in_features, out_perhead, _adj, dropout=dropout) for _ in range(heads)])

        self.in_features = in_features
        self.out_perhead = out_perhead
        self.heads = heads

    def forward(self, input, adj):
        output = torch.cat([att(input, adj) for att in self.graph_atts], dim=1)
        # notice that original GAT use elu as activation func. 
        return output

    def __repr__(self):
        return self.__class__.__name__ + "({}->[{}x{}])".format(
                    self.in_features, self.heads, self.out_perhead)
   
class GraphAttConvOneHead(nn.Module):
    """
    Sparse version GAT layer, single head
    """

    def __init__(self, in_features, out_features, _adj, dropout=0.6, alpha=0.2):
        super(GraphAttConvOneHead, self).__init__()

        self.weight = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        self._adj = _adj
        if self._adj != None:
            print("using new initailization")
            stdv = self.get_std()
            # print(stdv)
            for i in range(0, self.weight.shape[0]):
                self.weight[i].data.uniform_(-stdv, stdv)
            self.a.data.uniform_(-stdv, stdv)
        else:
            # init 
            nn.init.xavier_normal_(self.weight.data, gain=nn.init.calculate_gain('relu')) # look at here
            nn.init.xavier_normal_(self.a.data, gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def get_std(self):
        N = self._adj.shape[0]
        d_i = torch.sparse.sum(self._adj, dim = 1).to_dense().unsqueeze(1) + 1
        d_j = torch.sparse.sum(self._adj, dim = 0).to_dense().unsqueeze(0) + 1
        stdv = N/torch.sqrt(torch.sum(d_i * d_j) * 3)
        return stdv

    def forward(self, input, adj):
        edge = adj._indices()
        h = torch.mm(input, self.weight)
        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t() # edge_h: 2*D x E
        # do softmax for each row, this need index of each row, and for each row do softmax over it
        alpha = self.leakyrelu(self.a.mm(edge_h).squeeze()) # E
        n = len(input)
        alpha = softmax(alpha, edge[0], n)
        output = spmm(edge, self.dropout(alpha), n, n, h) # h_prime: N x out
        # output = spmm(edge, self.dropout(alpha), n, n, self.dropout(h)) # h_prime: N x out
        return output
    
class PairNorm(nn.Module):
    def __init__(self, mode='PN', scale=1):
        """
            mode:
              'None' : No normalization 
              'PN'   : Original version
              'PN-SI'  : Scale-Individually version
              'PN-SCS' : Scale-and-Center-Simultaneously version
           
            ('SCS'-mode is not in the paper but we found it works well in practice, 
              especially for GCN and GAT.)

            PairNorm is typically used after each graph convolution operation. 
        """
        assert mode in ['None', 'PN',  'PN-SI', 'PN-SCS']
        super(PairNorm, self).__init__()
        self.mode = mode
        self.scale = scale

        # Scale can be set based on origina data, and also the current feature lengths.
        # We leave the experiments to future. A good pool we used for choosing scale:
        # [0.1, 1, 10, 50, 100]
                
    def forward(self, x):
        if self.mode == 'None':
            return x
        
        col_mean = x.mean(dim=0)      
        if self.mode == 'PN':
            x = x - col_mean
            rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt() 
            x = self.scale * x / rownorm_mean

        if self.mode == 'PN-SI':
            x = x - col_mean
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual

        if self.mode == 'PN-SCS':
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual - col_mean

        return x

"""
    helpers
    # print("Using new initialization")
            # N = self.adj.shape[0]
            # d_i = torch.sparse.sum(self.adj, dim = 1).to_dense() + 1
            # d_j = torch.sparse.sum(self.adj, dim = 0).to_dense() + 1
            # T_ij = 0.0
            # for i in tqdm(range(0, N)):
            #     for j in range(0, N):
            #         h = 1/torch.sqrt(d_i[i] * d_j[j])
            #         k = 0.0
            #         for item1 in torch.where(self.adj[i].to_dense() != 0.0)[0]:
            #             for item2 in torch.where(self.adj[j].to_dense() != 0.0)[0]:
            #                 l = 1/torch.sqrt(d_i[item1] * d_j[item2])
            #                 k = k + l
            #         T_ij = T_ij + (h * k)
            #     print("{}\t{}\n".format(i, T_ij))
            #         # for m in range(0, N):
            #         #     for n in range(0, N):
            #         #         print(n)
            #         #         if self.adj[i][m] != 0 and self.adj[j][n] != 0:
            #         #             l = 1/torch.sqrt(d_i[m] * d_j[n])
            #         #             T_ij += h * l
            # stdv = N/torch.sqrt(T_ij * 3)
            # print(T_ij, stdv)
            # sys.exit(0)
            # for i in range(0, self.weight.shape[0]):
            #     self.weight[i].data.uniform_(-stdv, stdv)
            # if self.bias is not None:
            #     self.bias.data.uniform_(-stdv, stdv)

    d_i = torch.sparse.sum(self.adj, dim = 1).to_dense() + 1
        d_j = torch.sparse.sum(self.adj, dim = 0).to_dense() + 1
        T_ij = 0.0
        for i in tqdm(range(0, N)):
            for j in range(0, N):
                h = 1/(d_i[i] * d_j[j])
                k = 0.0
                for item1 in torch.where(self.adj[i].to_dense() != 0.0)[0]:
                    if item1 == i:
                        continue
                    for item2 in torch.where(self.adj[j].to_dense() != 0.0)[0]:
                        if item2 == j:
                            continue
                        l = 1/(d_i[item1] * d_j[item2])
                        k = k + l
                T_ij = T_ij + (h * k)
            print("{}\t{}\n".format(i, T_ij.cpu().numpy()))
        stdv = torch.sqrt((3 * N * N)/torch.sqrt(T_ij + (N * N)))
        print(T_ij, stdv.cpu().numpy())
        sys.exit(0)


        def get_std(self):
        N = self.adj.shape[0]
        d_i = torch.sparse.sum(self.adj, dim = 1).to_dense().unsqueeze(1) + 1
        d_j = torch.sparse.sum(self.adj, dim = 0).to_dense().unsqueeze(0) + 1
        stdv = N/torch.sqrt(torch.sum(d_i * d_j) * 3)
        return stdv
"""
from torch_scatter import scatter_max, scatter_add
def softmax(src, index, num_nodes=None):
    """
        sparse softmax
    """
    num_nodes = index.max().item() + 1 if num_nodes is None else num_nodes
    out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
    out = out.exp()
    out = out / (scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)
    return out
