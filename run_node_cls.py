import argparse

from models import GCNmf, GCNfse
from train import NodeClsTrainer
from utils import NodeClsData, apply_mask, generate_mask


parser = argparse.ArgumentParser()
parser.add_argument('--dataset',
                    default='cora',
                    choices=['cora', 'citeseer', 'amacomp', 'amaphoto'],
                    help='dataset name')
parser.add_argument('--type',
                    default='uniform',
                    choices=['uniform', 'bias', 'struct'],
                    help="uniform randomly missing, biased randomly missing, and structurally missing")
parser.add_argument('--rate', default=0.1, type=float, help='missing rate')
parser.add_argument('--nhid', default=16, type=int, help='the number of hidden units')
parser.add_argument('--dropout', default=0.5, type=float, help='dropout rate')
# parser.add_argument('--ncomp', default=5, type=int, help='the number of Gaussian components')
parser.add_argument('--lr', default=0.005, type=float, help='learning rate')
parser.add_argument('--wd', default=1e-2, type=float, help='weight decay')
parser.add_argument('--epoch', default=10000, type=int, help='the number of training epoch')
parser.add_argument('--patience', default=100, type=int, help='patience for early stopping')
parser.add_argument('--verbose', action='store_true', help='verbose')

parser.add_argument('--emb1', default=101, type=int, help='k : the size of linear combination')
parser.add_argument('--emb2', default=102, type=int, help='m : the size of rank refularization')
parser.add_argument('--emb3_1', default=103, type=int, help='la : the size of set embedding')
parser.add_argument('--emb3_2', default=103, type=int, help='lb : the size of set embedding')

args = parser.parse_args()

if __name__ == '__main__':
    data = NodeClsData(args.dataset)
    mask = generate_mask(data.features, args.rate, args.type)
    apply_mask(data.features, mask)
    model = GCNfse(data, nhid=args.nhid, dropout=args.dropout, n_emb1=args.emb1, n_emb2=args.emb2, n_emb3_1=args.emb3_1, n_emb3_2=args.emb3_2)
    params = {
        'lr': args.lr,
        'weight_decay': args.wd,
        'epochs': args.epoch,
        'patience': args.patience,
        'early_stopping': True
    }
    trainer = NodeClsTrainer(data, model, params, niter=20, verbose=args.verbose)
    trainer.run()
