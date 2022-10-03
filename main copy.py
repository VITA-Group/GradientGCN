import os, torch, logging, argparse
import rewire_model
from utils import train, test, val
from data import load_data
import shutil
import numpy as np
import random

# os.environ["CUDA_VISIBLE_DEVICES"] = '7'
use_cuda = torch.cuda.is_available()

def setup_seed(seed): 
    print('setup random seed = {}'.format(seed))
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed) 
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True 

# out dir 
OUT_PATH = "results/"
# if os.path.isdir(OUT_PATH):
#     shutil.rmtree(OUT_PATH)
# os.mkdir(OUT_PATH)
# parser for hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='citeseer', help='{cora, pubmed, citeseer}.')
parser.add_argument('--model', type=str, default='DeepGCN', help='{SGC, DeepGCN, DeepGAT}')
parser.add_argument('--hid', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--nhead', type=int, default=1, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate.')
parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train.')
parser.add_argument('--log', type=str, default='debug', help='{info, debug}')
parser.add_argument('--wd', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
# for deep model
parser.add_argument('--nlayer', type=int, default=14, help='Number of layers, works for Deep model.')
parser.add_argument('--residual', type=int, default=0, help='Residual connection')
# for PairNorm
# - PairNorm mode, use PN-SI or PN-SCS for GCN and GAT. With more than 5 layers get lots improvement.
parser.add_argument('--norm_mode', type=str, default='None', help='Mode for PairNorm, {None, PN, PN-SI, PN-SCS}')
parser.add_argument('--norm_scale', type=float, default=1.0, help='Row-normalization scale')
# for data
parser.add_argument('--no_fea_norm', action='store_false', default=True, help='not normalize feature' )
parser.add_argument('--missing_rate', type=int, default=0, help='missing rate, from 0 to 100' )
parser.add_argument('--new_init', type=int, default=1, help='use data-centric initialization' )
parser.add_argument('--seed', type=int, default=1, help='use data-centric initialization' )

args = parser.parse_args()

setup_seed(args.seed)
# logger
#filename='example.log'
logging.basicConfig(format='%(message)s', level=getattr(logging, args.log.upper())) 

# load data
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
# net = torch.nn.DataParallel(net)
net = net.cuda()
print(net)
optimizer = torch.optim.Adam(net.parameters(), args.lr, weight_decay=args.wd)
criterion = torch.nn.CrossEntropyLoss()
logging.info(net)

# train
best_acc = 0 
bes_test_acc = 0
best_loss = 1e10
for epoch in range(args.epochs):
    train_loss, train_acc, mean_grad = train(net, optimizer, criterion, data)
    val_loss, val_acc = val(net, criterion, data)
    test_loss, test_acc = test(net, criterion, data)
    if epoch % 1 == 0:
        print('Epoch %d: train loss %.3f train acc: %.3f, val loss: %.3f val acc %.3f, test loss: %.3f test acc %.3f mean grad %.5f'%
                (epoch, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc, mean_grad))
    # save model 
    if best_acc < val_acc:
        best_acc = val_acc
        bes_test_acc = test_acc
#         torch.save(net.state_dict(), OUT_PATH+'checkpoint-best-acc.pkl')
#     # if best_loss > val_loss:
#     #     best_loss = val_loss
#     #     torch.save(net.state_dict(), OUT_PATH+'checkpoint-best-loss.pkl')

# # pick up the best model based on val_acc, then do test

# net.load_state_dict(torch.load(OUT_PATH+'checkpoint-best-acc.pkl'))
# val_loss, val_acc = val(net, criterion, data)
# test_loss, test_acc = test(net, criterion, data)

# fopen = open("gcn_result.txt", "a")
# # if args.nlayer == 2:
# #     fopen.write("Layer\tval_loss\tval_acc\ttest_loss\ttest_acc\n")
print("-"*50)
print("Best Val Acc : {:.3f} and Best Test Acc : {:.3f}".format(best_acc, bes_test_acc))
# print("Vali set results: loss %.3f, acc %.3f."%(val_loss, val_acc))
# print("Test set results: loss %.3f, acc %.3f."%(test_loss, test_acc))
print("="*50)

# fopen.write("{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n".format(args.nlayer, val_loss, val_acc, test_loss, test_acc))
# fopen.flush()
# fopen.close()