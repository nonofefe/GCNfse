import argparse

from models import GCNmf
from train import NodeClsTrainer
from utils import NodeClsData, apply_mask, generate_mask
import torch

if __name__ == '__main__':
    data = torch.zeros(10,10)
    data += 0.1
    print(data)
    mask = generate_mask(data, 0.1, "uniform")
    print(mask)
    # apply_mask(data.features, mask)
    # model = GCNmf(data, nhid=args.nhid, dropout=args.dropout, n_components=args.ncomp)
    # params = {
    #     'lr': args.lr,
    #     'weight_decay': args.wd,
    #     'epochs': args.epoch,
    #     'patience': args.patience,
    #     'early_stopping': True
    # }
    # trainer = NodeClsTrainer(data, model, params, niter=20, verbose=args.verbose)
    # trainer.run()
