import torch
import torch.nn as nn

from GraphConv import GraphConvolution


class GraphConvAutoEncoder_VertexRecon(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GraphConvAutoEncoder_VertexRecon, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, 2 * nhid)
        self.dropout = nn.Dropout(p=dropout)
        self.gc3 = GraphConvolution(2 * nhid, nclass)

    def forward(self, x, adj):
        x = torch.sigmoid(self.gc1(x, adj))
        x = torch.sigmoid(self.gc2(x, adj))
        feat = x
        x = self.dropout(x)
        x = self.gc3(x, adj)
        return x, feat


class GraphConvAutoEncoder_EdgeRecon(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GraphConvAutoEncoder_EdgeRecon, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, 2 * nhid)
        # self.dropout = nn.Dropout(p=dropout)
        # self.gc3 = GraphConvolution(2 * nhid, nclass)

    def forward(self, x, adj):
        x = torch.sigmoid(self.gc1(x, adj))
        x = torch.sigmoid(self.gc2(x, adj))
        # x = self.dropout(x)
        # x = self.gc3(x, adj)
        return x
