import math

import torch
import torch.nn.functional as F
from torch import nn

from models.GCNII_DenseLayer import GCNIIConv_arxiv
from models.GCNII_layer import GCNIIdenseConv
from utils import tonp, get_laplacian_mat, energy

class GCNII(nn.Module):
    def __init__(self, args):
        super(GCNII, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.cached = self.transductive = args.transductive

        gcn_conv = GCNIIConv_arxiv if self.dataset == 'ogbn-arxiv' else GCNIIdenseConv

        self.convs = torch.nn.ModuleList()
        self.convs.append(torch.nn.Linear(self.num_feats, self.dim_hidden))
        for _ in range(self.num_layers):
            self.convs.append(gcn_conv(self.dim_hidden, self.dim_hidden))

        self.convs.append(torch.nn.Linear(self.dim_hidden, self.num_classes))
        self.reg_params = list(self.convs[1:-1].parameters())
        self.non_reg_params = list(self.convs[0:1].parameters()) + list(self.convs[-1:].parameters())

        self.optimizer = torch.optim.Adam([
            dict(params=self.reg_params, weight_decay=self.weight_decay1),
            dict(params=self.non_reg_params, weight_decay=self.weight_decay2)
        ], lr=self.lr)

    def forward(self, x, edge_index):
        L = get_laplacian_mat(edge_index,  x.shape[0])
        fopen = open("./runs/data_gcnII_l16_cora__wo_identity.txt", "a")
        
        _hidden = []
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.convs[0](x)
        x = F.relu(x)
        e = energy(tonp(x.clone()), L)
        fopen.write("{:.3f}\t".format(e))
        x_init = x
        x_last = x

        for i, con in enumerate(self.convs[1:-1]):
            x = F.dropout(x, self.dropout, training=self.training)
            if self.dataset != 'ogbn-arxiv':
                beta = math.log(self.lamda / (i + 1) + 1)
                x = F.relu(con(x, edge_index, self.alpha, x_init, beta))
            else:
                x = F.relu(con(x, edge_index, self.alpha, x_init)) + x_last
                x_last = x
            e = energy(tonp(x.clone()), L)
            fopen.write("{:.3f}\t".format(e))
        fopen.write("\n")
        fopen.flush()
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.convs[-1](x)
        return x
