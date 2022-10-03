import os, torch, logging, argparse
import model
from utils import train, test, val, dynamic_rewire
from data import load_data
import shutil
import numpy as np
import random
import sys
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

# parser for hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='cora', help='{cora, pubmed, citeseer}.')
parser.add_argument('--model', type=str, default='DeepGCN', help='{SGC, DeepGCN, DeepGAT}')
parser.add_argument('--hid', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--nhead', type=int, default=1, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate.')
parser.add_argument('--epochs', type=int, default=3500, help='Number of epochs to train.')
parser.add_argument('--log', type=str, default='debug', help='{info, debug}')
parser.add_argument('--wd', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
# for deep model
parser.add_argument('--nlayer', type=int, default=10, help='Number of layers, works for Deep model.')
parser.add_argument('--residual', type=int, default=0, help='Residual connection')
# for PairNorm
# - PairNorm mode, use PN-SI or PN-SCS for GCN and GAT. With more than 5 layers get lots improvement.
parser.add_argument('--norm_mode', type=str, default='None', help='Mode for PairNorm, {None, PN, PN-SI, PN-SCS}')
parser.add_argument('--norm_scale', type=float, default=1.0, help='Row-normalization scale')
# for data
parser.add_argument('--no_fea_norm', action='store_false', default=True, help='not normalize feature' )
parser.add_argument('--missing_rate', type=int, default=0, help='missing rate, from 0 to 100' )
parser.add_argument('--new_init', type=int, default=0, help='use data-centric initialization' )
parser.add_argument('--rewire', type=int, default=1, help='use data-centric initialization' )
parser.add_argument('--seed', type=int, default=1, help='use data-centric initialization' )


def main():
    args = parser.parse_args()
    print(args)
    setup_seed(args.seed)

    # rewiring_skip = [2,3,4,5,6,7,8,9]
    rewiring_skip = []
    if args.rewire == 1:
        rewiring_skip = dynamic_rewire(args)
        print("Rewiring Skip List : {}".format(rewiring_skip))

  

    setup_seed(args.seed + 2)

    dataset = load_data(args.data, normalize_feature=args.no_fea_norm, missing_rate=args.missing_rate, cuda=True)
    nfeat = dataset.x.size(1)
    nclass = int(dataset.y.max()) + 1
    if args.new_init == 1:
        adj = dataset.adj.clone()
    else:
        adj = None
    new_model = getattr(model, "DeepGCNII")(nfeat, args.hid, nclass, 
                                      dropout=args.dropout, 
                                      nhead=args.nhead,
                                      nlayer=args.nlayer, 
                                      adj = adj,
                                      norm_mode=args.norm_mode,
                                      norm_scale=args.norm_scale,
                                      residual=args.residual)
    new_model.skip_list = rewiring_skip
    new_model.cuda()
    print(new_model)
    # # train
    optimizer = torch.optim.Adam(new_model.parameters(), args.lr, weight_decay=args.wd)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.0)
    best_acc = 0 
    bes_test_acc = 0
    best_overall = 0
    best_loss = 1e10
    
    # fopen = open("./plots/layer_grad_data_skip_new.txt", "a")
    for epoch in range(args.epochs):
        train_loss, train_acc, mean_grad, _ = train(new_model, optimizer, criterion, dataset)
        val_loss, val_acc = val(new_model, criterion, dataset)
        test_loss, test_acc = test(new_model, criterion, dataset)
        if epoch % 1 == 0:
            print('Epoch %d: train loss %.3f train acc: %.3f, val loss: |%.3f| val acc %.3f, test loss: %.3f test acc %.3f mean grad ||%.5f'%
                    (epoch, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc, np.mean(mean_grad)))
            # fopen.write("\t".join(["{:.3f}".format(i) for i in mean_grad]))
            # fopen.write("\n")
            # fopen.flush()
            # fopen.write('Epoch %d: train loss %.3f train acc: %.3f, val loss: |%.3f| val acc %.3f, test loss: %.3f test acc %.3f mean grad ||%.5f'%
            #         (epoch, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc, np.mean(mean_grad)))
            # fopen.write("\n")
            # fopen.flush()
        # save model 
        if best_acc < val_acc:
            best_acc = val_acc
            bes_test_acc = test_acc
        if best_overall < test_acc:
            best_overall = test_acc
    
    fopen = open("data.txt", "a")
    fopen.write("Data: {} || Layer : {} || Skip List : {} Best Val Acc : {:.4f} and Best Test Acc : {:.4f} and Best Overall : {:.4f}\n".format(args.data, args.nlayer, rewiring_skip, best_acc, bes_test_acc, best_overall))
    print("Data: {} || Layer : {} || Skip List : {} Best Val Acc : {:.4f} and Best Test Acc : {:.4f} and Best Overall : {:.4f}\n".format(args.data, args.nlayer, rewiring_skip, best_acc, bes_test_acc, best_overall))
    fopen.close()

if __name__ == '__main__':
    main()

#Cora seed - 99