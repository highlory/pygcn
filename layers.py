import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    # 初始化待学习的参数
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)  # 使用均匀分布随机初始化权重
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)  # torch.spmm()仅支持sparse与dense的乘法，即第一个参数为稀疏矩阵，第二个参数为普通的矩阵
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    
    def show_details(self):
        print('weight:')
        print(self.weight)
        if self.bias is not None:
            print('bias:')
            print(self.bias)
        print()

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

if __name__ == "__main__":
    GCN_layer = GraphConvolution(in_features=5, out_features=3)
