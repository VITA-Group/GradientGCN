import torch
from torch_geometric.utils import get_laplacian
import numpy as np
from layers import *
import  rewire_model
from data import load_data
import shutil
import numpy as np
import random
import os, torch, logging, argparse
from tqdm import tqdm

def train(net, optimizer, criterion, data):
    net.train()
    optimizer.zero_grad()
    output, skip_list = net(data.x, data.adj, data.edge_index, False)
    loss = criterion(output[data.train_mask], data.y[data.train_mask])
    acc = accuracy(output[data.train_mask], data.y[data.train_mask])
    loss.backward()
    grad_norm = 0.0
    layer_grad = []
    layer_count = 0
    for name, m in net.named_modules():
        if isinstance(m, GraphConv):
            grad_norm += float(torch.norm(m.weight.grad.clone()).cpu())
            layer_grad.append(float(torch.norm(m.weight.grad.clone()).cpu()))
            layer_count += 1
    grad_flow = grad_norm / layer_count
    # scores = {}
    # for name, m in net.named_modules():
    #     if isinstance(m, GraphConv):
    #         scores[name] = torch.clone(m.weight.grad.clone()).detach()
    # all_scores = torch.cat([torch.flatten(v) for v in scores.values()])
    # grad_flow = torch.norm(all_scores)
    optimizer.step()
    return loss, acc, layer_grad, skip_list 

def val(net, criterion, data):
    net.eval()
    output, _ = net(data.x, data.adj, data.edge_index, True)
    loss_val = criterion(output[data.val_mask], data.y[data.val_mask])
    acc_val = accuracy(output[data.val_mask], data.y[data.val_mask])
    return loss_val, acc_val

def test(net, criterion, data):
    net.eval()
    output, _ = net(data.x, data.adj, data.edge_index, True)
    loss_test = criterion(output[data.test_mask], data.y[data.test_mask])
    acc_test = accuracy(output[data.test_mask], data.y[data.test_mask])
    return loss_test, acc_test

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def dynamic_rewire(args):
    data = load_data(args.data, normalize_feature=args.no_fea_norm, missing_rate=args.missing_rate, cuda=True)
    nfeat = data.x.size(1)
    nclass = int(data.y.max()) + 1

    if args.new_init == 1:
        adj = data.adj.clone()
    else:
        adj = None
    net = getattr(rewire_model, args.model)(nfeat, args.hid, nclass, 
                                    dropout=args.dropout, 
                                    nhead=args.nhead,
                                    nlayer=args.nlayer, 
                                    adj = adj,
                                    norm_mode=args.norm_mode,
                                    norm_scale=args.norm_scale,
                                    residual=args.residual)
    net = net.cuda()
    print(net)
    optimizer = torch.optim.Adam(net.parameters(), args.lr, weight_decay=args.wd)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(200):
        train_loss, train_acc, mean_grad, skip_list = train(net, optimizer, criterion, data)
    
    return skip_list