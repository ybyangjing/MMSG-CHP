
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLError
from dgl.nn.pytorch import ChebConv, GraphConv
from model.auto_correlation import AutoCorrelationLayer,AutoCorrelation






class Gatconv(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats, n_heads=1):
        super(Gatconv, self).__init__()
        self.layer1 = GraphConv(in_feats, hidden_size, activation=F.relu, allow_zero_in_degree=True)
        self.auto_corr = AutoCorrelationLayer(AutoCorrelation(), hidden_size, n_heads)
        self.layer2 = GraphConv(hidden_size, out_feats, allow_zero_in_degree=True)


    def forward(self, g, node_feats, edge_feats=None):
        # Apply initial GraphConv layer
        if g.number_of_nodes() == 0:  
            h = self.layer1(g, node_feats)
            h = self.layer2(g, h)
            return h
        h = self.layer1(g, node_feats)

        # Prepare inputs for the AutoCorrelation Layer
        queries = keys = values = h.unsqueeze(0)  # Add batch dimension
        h, attn = self.auto_corr(queries, keys, values, None)
        h = h.squeeze(0)  # Remove batch dimension

        # Final GraphConv layer
        h = self.layer2(g, h)
        return h


def max_reduce(nodes):
    return {'h': torch.max(nodes.mailbox['e'], dim=1)[0]}


class ERConv(nn.Module):
    def __init__(self,
                 in_feat,
                 out_feat,
                 allow_zero_in_degree=True,
                 activation=True):
        super(ERConv, self).__init__()
        self._allow_zero_in_degree = allow_zero_in_degree
        self.activation = activation
        self.theta = nn.Linear(in_feat, out_feat)
        self.phi = nn.Linear(in_feat, out_feat)

    def message(self, edges):

        theta_x = self.theta((edges.src['h'] + edges.dst['h']) * edges.data['d'].view((-1, 1)))
        return {'e': theta_x}

    def forward(self, g, feat):
        with g.local_scope():
            if not self._allow_zero_in_degree:
                if (g.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph')
            g = g.to("cuda:0")
            g.ndata['h'] = feat[0]
            g.edata['d'] = feat[1]
            g.update_all(self.message, max_reduce)

            return dgl.max_nodes(g, 'h')


class GATRE(nn.Module):
    def __init__(self,in_feats, hidden_size, out_feats,drop):
        super(GATRE, self).__init__()
        self.Gatconv = Gatconv(in_feats, hidden_size, hidden_size)
        self.ERConv = ERConv(hidden_size, out_feats, activation=F.relu)
        self.dropout = nn.Dropout(drop)

    def forward(self, g, inputs):
        g=g.to("cuda:0")
        node_feats = inputs[0]

        h = self.Gatconv(g, node_feats)
        h = F.relu(h)
        h = self.ERConv(g, [h, inputs[-1]])
        h = self.dropout(h)

        return h
