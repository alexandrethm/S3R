import itertools
from enum import Enum

import torch
from torch import nn
import numpy as np


# Custom modules

class Swish(nn.Module):
    """Applies element-wise the function
     Swish(x) = x·Sigmoid(βx).


    Here :math:`β` is a learnable parameter. When called without arguments, `Swish()` uses a single
    parameter :math:`β` across all input channels. If called with `Swish(nChannels)`,
    a separate :math:`β` is used for each input channel.

    Args:
        num_parameters: number of :math:`β` to learn. Default: 1
        init: the initial value of :math:`β`. Default: 0.25

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input


    Examples::
        >>> m = Swish()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def __init__(self, num_parameters=1, init=0.25):
        self.num_parameters = num_parameters
        super(Swish, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_parameters).fill_(init))

    def forward(self, input):
        return input * torch.sigmoid(self.weight * input)

    def extra_repr(self):
        return 'num_parameters={}'.format(self.num_parameters)


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

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
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
    Transforms a (_, _, 22, f) batch (that went through a graph convolution) into a list of inputs for the following
    pipelines : [22 * (batch_size, f, temporal_duration)]
    """

    def forward(self, x):
        x = list(x.unbind(dim=2))
        for j in range(len(x)):
            # x[j] : (batch_size, temporal_duration, f)
            # with f : nb of features per node, after graph convolutions
            _, _, f = x[j].size()

            # x[j] : [f * (batch_size, 1, temporal_duration)]
            x[j] = [x[j][:, :, i].unsqueeze(dim=1) for i in range(f)]
            # x[j] : (batch_size, f, temporal_duration)
            x[j] = torch.cat(x[j], dim=1)

        return x


class GCN(nn.Module):
    def __init__(self, f):
        super().__init__()

        # input : (batch_size, temporal_duration, nb_sequences)
        self.shape = ShapeGraph()
        # GraphConv : 3 features per node/joint -> f features per node/joint
        self.graph_conv = GraphConvolution(3, f)
        # output : [22 * (batch_size, f, temporal_duration)]
        self.unshape = UnshapeGraph()

    def forward(self, input, adj):
        input = self.shape(input)
        input = self.graph_conv(input, adj)
        input = self.unshape(input)
        return input


# Different net types

class NetType(Enum):
    xyz = 'xyz'
    '''
    'xyz' for a XYZ network, with 3 pipelines (one for x_i sequences, one for y_i sequences, one for z_i sequences)
    '''

    regular = 'regular'
    '''
    'regular' for a network with 1 sequence per pipeline
    '''

    LSC = 'LSC'
    '''
    'LSC' (linear spatial combination) for a linear combination to be applied on the sequences before the conv/pool
    pipelines, in order to mix them.

    The same linear transformation is applied on all the sequences at each time step.

        input_batch : (batch_size, temporal_duration, initial_nb_sequences=66) ->
        preprocessed_batch : (batch_size, temporal_duration, nb_sequences=m*n)

    With m*n (nb_seq_per_pipeline*nb_pipelines, nb of features out of the linear combination)
     not necessarily the initial nb of sequences.
    '''

    graph_conv = 'graph_conv'
    '''
    'graph_conv' for a graph convolution on the joints before the conv/pool pipelines.
    
        input_batch : (batch_size, temporal_duration, initial_nb_sequences=66),
         seen as a graph (each joint being a node with 3 features : x_i, y_i and z_i) ->
        preprocessed_batch : (batch_size, temporal_duration, nb_sequences=22*k)

    With k the number of output features per node. After that, the features will be fed to 22 pipelines (one pipeline
    per node/joint), with l features per pipeline.
    '''


def xavier_init(layer, activation_fct):
    param = None

    if activation_fct == 'prelu':
        activation_fct = 'leaky_relu'
        param = torch.nn.PReLU().weight.item()
    elif activation_fct == 'swish':
        activation_fct = 'sigmoid'

    torch.nn.init.xavier_uniform_(layer.weight, gain=torch.nn.init.calculate_gain(activation_fct, param=param))
    torch.nn.init.constant_(layer.bias, 0.1)


def num_flat_features(x):
    """
    :param x: An input tensor, generally already processed with conv layers
    :return: The number of flat features of this tensor (except the batch dimension), that can be fed to a fully connected layer.
    """
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


def perform_xavier_init(module_lists, modules, activation_fct):
    """
    Perform xavier_init on Conv1d and Linear layers insides the specified modules.
    :param module_lists: list of ModuleList objects
    :param modules: list of Module objects
    :param activation_fct:
    """
    for module in itertools.chain(module_lists):
        for layer in module:
            if layer.__class__.__name__ == "Conv1d" or layer.__class__.__name__ == "Linear":
                xavier_init(layer, activation_fct)

    for layer in itertools.chain(modules):
        if layer.__class__.__name__ == "Conv1d" or layer.__class__.__name__ == "Linear":
            xavier_init(layer, activation_fct)


def get_preprocess_module(net_type, net_shape):
    if net_type is NetType.LSC.value:
        m, n = net_shape
        return nn.Linear(66, m * n)

    elif net_type is NetType.graph_conv.value:
        f, _ = net_shape
        return GCN(f)
    else:
        return None


def get_network_shape(net_type, net_shape):
    """
    :param net_type: str
    :param net_shape:
        For `linear_combination` networks : (nb_seq_per_pipeline, nb_pipelines)
        For `graph_conv` networks : (nb_features_per_node)
        Otherwise it's already determined by the network type.
    :return: m (number of of channels per pipeline), n (number of pipelines).
    """
    if net_type is NetType.xyz.value:
        return 22, 3
    elif net_type is NetType.regular.value:
        return 1, 66
    elif net_type is NetType.LSC.value:
        return net_shape
    elif net_type is NetType.graph_conv.value:
        return net_shape


def get_pipeline_inputs(input_batch, net_type, net_shape):
    """
    Get inputs in the right shape, according to the network type specified.

    :param input_batch: A tensor of gestures. Its shape is (batch_size, temporal_duration, nb_sequences)
    :param net_type: str
    :param net_shape: Network shape (m, n) if required (for networks with pre-processing for instance)
    :return: A list containing the inputs for each pipeline. Its shape is [n x (batch_size, m, temporal_duration)]
    With each of the n pipelines fed with m sequences, it is NECESSARY to have m x n = nb_sequences, where nb_sequences
    is the total number of sequences provided as input.

    Note that we are talking about (possibly) preprocessed input and sequences, so nb_sequences is not necessarily
    the initial nb of sequences (66 for the SHREC data).
    """
    _, _, nb_sequences = input_batch.size()
    m, n = get_network_shape(net_type, net_shape)
    # make sure the dimensions are right
    assert m * n == nb_sequences, 'Number of sequences in batch input does not match network shape'

    pipeline_inputs = []

    if net_type is NetType.xyz.value:
        for i in range(n):
            # Get all x_i (or y_i or z_i) time sequences in a list,
            # each one with the following shape : (batch_size, 1, temporal_duration)
            pipeline_input = [input_batch[:, :, 3 * j + i].unsqueeze(1) for j in range(m)]

            # Concatenate the list to get an appropriate shape : (batch_size, m=22, temporal_duration),
            # so it fits Conv1D format : (batch_size, num_feature_maps, length_of_seq)
            pipeline_input = torch.cat(pipeline_input, 1)

            pipeline_inputs.append(pipeline_input)
        return pipeline_inputs

    elif net_type is NetType.regular.value:
        # Get all time sequences in a list,
        # each one with the following shape : (batch_size, m=1, temporal_duration)
        # so it fits Conv1D format : (batch_size, num_feature_maps, length_of_seq)
        pipeline_inputs = [input_batch[:, :, j].unsqueeze(1) for j in range(n)]

        return pipeline_inputs

    elif net_type is NetType.LSC.value:
        # Split the input batch into a list of n pipeline_input
        for i in range(n):
            # Add m time sequences to the input,
            # each one with the following shape : (batch_size, 1, temporal_duration)
            pipeline_input = [input_batch[:, :, i * m + j].unsqueeze(1) for j in range(m)]

            # Concatenate the list to get an appropriate shape : (batch_size, m, temporal_duration),
            # so it fits Conv1D format : (batch_size, num_feature_maps, length_of_seq)
            pipeline_input = torch.cat(pipeline_input, 1)

            pipeline_inputs.append(pipeline_input)
        return pipeline_inputs


# Adjacency matrix of the hand graph
adj = torch.tensor(
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
