import numpy as np
import torch

from torch import nn


class AdjacencyMatrix(object):
    hand = torch.tensor(
        [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
         [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]],
        dtype=torch.float
    )
    if torch.cuda.is_available():
        hand = hand.cuda()


class GCN(nn.Module):
    """
    A graph convolution layer. The sequences are interpreted as the features of a graph's nodes.
    For example, with 66 sequences for the hand movement (SHREC 2017 dataset), the graph will have 22 nodes of
    3 features each (x_i, y_i, z_i).


    Shape
        - Input : :math:`(N, L, C_{in})`, with C_in = nb_nodes * nb_features_in_per_node
        - Output : :math:`(N, L, C_{out})` where C_out = nb_nodes * nb_features_out_per_node

    Args:
        output_channels (int): C_out, must be divisible by the number of nodes of the graph

    """

    def __init__(self, output_channels):
        super(GCN, self).__init__()
        if output_channels % 22 != 0:
            raise AttributeError('Wrong parameter specified for output_channels ({}), it must be ' \
                                          'divisible by the number of nodes of the graph'.format(output_channels))
        f = int(output_channels / 22)

        # input : (batch_size, temporal_duration, nb_sequences)
        self.shape = ShapeGraph()
        # GraphConv : 3 features per node/joint -> f features per node/joint
        self.graph_conv = GraphConvolution(3, f)
        # output : [22 * (batch_size, f, temporal_duration)]
        self.unshape = UnshapeGraph()

    def forward(self, x, adj=AdjacencyMatrix.hand):
        x = self.shape(x)
        x = self.graph_conv(x, adj)
        x = self.unshape(x)
        return x


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        """
        Args:
            x: Input with shape (_, _, N, F) where N is the number of nodes in the graph and F the number
                of features per node
            adj: Adjacency matrix of the graph, size (N, N)
        """
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class ShapeGraph(nn.Module):
    """
    Transforms a (_, _, 66) input_batch into a (_, _, 22, 3) batch to separate sequences by joint. Each joint
    has its 3 coordinates.
    """

    def forward(self, x):
        return torch.stack(
            tensors=x.chunk(chunks=22, dim=2),
            dim=2
        )


class UnshapeGraph(nn.Module):
    """
    Transforms a (_, _, 22, f) batch (that went through a graph convolution) into a (_, _, 22*f) batch
    """

    def forward(self, x):
        return torch.cat(
            [x[:, :, i, :] for i in range(x.shape[2])],
            dim=2
        )
