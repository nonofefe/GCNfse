import torch.nn as nn
import torch.nn.functional as F

from . import GCNConv, GCNmfConv, FSE#, GCN, FSE


class GCNfse(nn.Module):
    def __init__(self, data, nhid=16, dropout=0.5, n_emb1=4, n_emb2=4, n_emb3=4):
        super(GCNfse, self).__init__()
        nfeat, nclass = data.num_features, data.num_classes
        self.fse = FSE(nfeat, n_emb1, n_emb2, n_emb3, data, dropout)
        self.gc1 = GCNConv(n_emb1, nhid, dropout)
        self.gc2 = GCNConv(nhid, nclass, dropout)

    def reset_parameters(self):
        self.fse.reset_parameters()
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def forward(self, data):
        x, adj = data.features, data.adj
        # FSE
        x = self.fse(x,adj)
        # GCN
        x = F.relu(self.gc1(x, adj))
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
