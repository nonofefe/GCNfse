import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from torch.nn.parameter import Parameter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def ex_relu(mu, sigma):
    is_zero = (sigma == 0)
    sigma[is_zero] = 1e-10
    sqrt_sigma = torch.sqrt(sigma)
    w = torch.div(mu, sqrt_sigma)
    nr_values = sqrt_sigma * (torch.div(torch.exp(torch.div(- w * w, 2)), np.sqrt(2 * np.pi)) +
                              torch.div(w, 2) * (1 + torch.erf(torch.div(w, np.sqrt(2)))))
    nr_values = torch.where(is_zero, F.relu(mu), nr_values)
    return nr_values


def init_gmm(features, n_components):
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    init_x = imp.fit_transform(features)
    gmm = GaussianMixture(n_components=n_components, covariance_type='diag').fit(init_x)
    return gmm


class GCNmfConv(nn.Module):
    def __init__(self, in_features, out_features, data, n_components, dropout, bias=True):
        super(GCNmfConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_components = n_components
        self.dropout = dropout
        self.features = data.features.numpy()
        self.logp = Parameter(torch.FloatTensor(n_components))
        self.means = Parameter(torch.FloatTensor(n_components, in_features))
        self.logvars = Parameter(torch.FloatTensor(n_components, in_features))
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.adj2 = torch.mul(data.adj, data.adj).to(device)
        self.gmm = None
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        if self.bias is not None:
            self.bias.data.fill_(0)
        self.gmm = init_gmm(self.features, self.n_components)
        self.logp.data = torch.FloatTensor(np.log(self.gmm.weights_)).to(device)
        self.means.data = torch.FloatTensor(self.gmm.means_).to(device)
        self.logvars.data = torch.FloatTensor(np.log(self.gmm.covariances_)).to(device)

    def calc_responsibility(self, mean_mat, variances):
        dim = self.in_features
        log_n = (- 1 / 2) *\
            torch.sum(torch.pow(mean_mat - self.means.unsqueeze(1), 2) / variances.unsqueeze(1), 2)\
            - (dim / 2) * np.log(2 * np.pi) - (1 / 2) * torch.sum(self.logvars)
        log_prob = self.logp.unsqueeze(1) + log_n
        return torch.softmax(log_prob, dim=0)

    def forward(self, x, adj):
        x_imp = x.repeat(self.n_components, 1, 1)
        x_isnan = torch.isnan(x_imp)
        variances = torch.exp(self.logvars)
        mean_mat = torch.where(x_isnan, self.means.repeat((x.size(0), 1, 1)).permute(1, 0, 2), x_imp)
        var_mat = torch.where(x_isnan,
                              variances.repeat((x.size(0), 1, 1)).permute(1, 0, 2),
                              torch.zeros(size=x_imp.size(), device=device, requires_grad=True))

        # dropout
        dropmat = F.dropout(torch.ones_like(mean_mat), self.dropout, training=self.training)
        mean_mat = mean_mat * dropmat
        var_mat = var_mat * dropmat

        transform_x = torch.matmul(mean_mat, self.weight)
        if self.bias is not None:
            transform_x = torch.add(transform_x, self.bias)
        transform_covs = torch.matmul(var_mat, self.weight * self.weight)
        conv_x = []
        conv_covs = []
        for component_x in transform_x:
            conv_x.append(torch.spmm(adj, component_x))
        for component_covs in transform_covs:
            conv_covs.append(torch.spmm(self.adj2, component_covs))
        transform_x = torch.stack(conv_x, dim=0)
        transform_covs = torch.stack(conv_covs, dim=0)
        expected_x = ex_relu(transform_x, transform_covs)

        # calculate responsibility
        gamma = self.calc_responsibility(mean_mat, variances)
        expected_x = torch.sum(expected_x * gamma.unsqueeze(2), dim=0)
        return expected_x

class FSE(nn.Module):
    def __init__(self, d, k, m, la, lb, data, dropout, bias=True):
        super(FSE, self).__init__()
        l = la + lb
        self.l = la+lb
        self.in_features = d
        self.k = k
        self.m = m
        self.la = la
        self.lb = lb
        self.dropout = dropout
        self.weight_V = Parameter(torch.FloatTensor(k,l))
        #self.weight_W = Parameter(torch.FloatTensor(m,l))
        self.weight_L = Parameter(torch.FloatTensor(l,d))

        self.features = data.features.numpy()
        self.features_nan = np.isnan(self.features)
        self.not_nan = np.count_nonzero(self.features_nan == False, axis=1)
        self.features[self.features_nan] = 0
        
        self.not_nan = 1 / (self.not_nan + 0.0001)
        #print(self.not_nan.shape)
        self.not_nan = np.tile(self.not_nan,(l,1))
        #print(self.not_nan.shape)
        self.not_nan = torch.from_numpy(self.not_nan).float()

        self.features = torch.from_numpy(self.features)
        self.features = self.features.T # 転置

        self.not_nan *= 1000
        self.not_nan.to(device)

        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.xavier_uniform_(self.weight_V.weight, gain=1.414)
        # nn.init.xavier_uniform_(self.weight_W.weight, gain=1.414)
        # nn.init.xavier_uniform_(self.weight_L.weight, gain=1.414)
        # if self.bias is not None:
        #     self.bias.data.fill_(0)

        nn.init.xavier_uniform_(self.weight_V.data, gain=1.414)
        #nn.init.xavier_uniform_(self.weight_W.data, gain=1.414)
        nn.init.xavier_uniform_(self.weight_L.data, gain=1.414)
        
    def forward(self, x, adj):
        feat = F.dropout(self.features, p=self.dropout, training=self.training)
        feat = feat.to(device)

        x = self.weight_V
        y = torch.matmul(self.weight_L, feat)
        y = y.to(device)
        not_nan = self.not_nan.to(device)
        y = torch.mul(y, not_nan)
        #y = F.dropout(y, p=self.dropout, training=self.training) #
        z = torch.matmul(x, y)
        z = torch.t(z)

        # #に変更
        #z = torch.sigmoid(z)
        #rowsum = z.sum(dim=1, keepdim=True)
        #z = z / rowsum



        # z = F.relu(z)
        
        # z = z / rowsum

        #z[z<0] = 0
        #z = F.softmax(z, dim=1)
        #rowsum = z.sum(dim=1, keepdim=True)
        #print(rowsum[0,0])

        # rowsum = z.sum(dim=1, keepdim=True)
        # #print(rowsum)
        # rowsum[rowsum == 0] = 1
        # z = z / rowsum
        return z