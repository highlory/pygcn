import torch.nn as nn
import torch.nn.functional as F
# from pygcn.layers import GraphConvolution
from layers import GraphConvolution


class GCN(nn.Module):
    # nfeat：第一层输入的维度；nhid：中间层的维度；nclass：输出层的维度
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
