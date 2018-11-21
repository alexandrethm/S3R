import torch
from torch import nn

from code_S3R.modules.activation_fct import Swish
from code_S3R.modules.fully_connected import FullyConnected
from code_S3R.modules.regular_cn import RegularConvNet
from code_S3R.modules.tcn import TemporalConvNet
from code_S3R.my_utils import training_utils
from code_S3R.modules.gcn import GCN


class Net(nn.Module):
    """
    Generic and modular class for temporal sequences classification. The structure is as follows :

    Input = (N, L_in, C_in) tensor, where
        - N is the batch size
        - L_in the length of the sequences
        - C_in the number of channels

    1. Preprocessing module
        - 'LSC' (linear spatial combination) for a linear combination to be applied on the sequences before the conv/pool
          pipelines, in order to mix them.

          The same linear transformation is applied on all the sequences at each time step.

          Shape
            input_batch : (N, L_in, C_in) ->
            preprocessed_batch : (N, L_in, C_preprocessed)
        - 'graph_conv' for a graph convolution on the joints before the
          conv/pool pipelines.

          The input is seen as a graph (each joint of the hand being a node with f_in=3 features : x_i, y_i and z_i).
          The graph is transformed into a new graph with the same number of nodes, but a (possibly) different number
          of features per node (f_out).

          Shape
            input_batch : (N, L_in, C_in = nb_nodes * f_in) ->
            preprocessed_batch : (N, L_in, C_preprocessed = nb_nodes * f_out)
        - None for no preprocessing module

    2. Convolution module
        - 'temporal'
        - 'regular

    3. Classification module (fully connected layers)

    todo : Add the attention modules


    Args:
        preprocess (str): Specify the preprocessing module, to be applied before the convolutions.
        conv_type (str): Specify the type of convolutions to use.
        channel_list (list of tuples): Specify the number of output channels of preprocessing/convolution layers, and
          the associated parameter (None for preprocessing, and groups for convolutions).
          If there is a preprocess module, the first tuple of the list will specify its output channels: (C, None)
          where C is the number of output channels. The rest of the list will be for the convolutions layers.
          e.g.
            >>> channel_list = [(C, None), (C_conv1, G_conv1), (C_conv2, G_conv2), ...]
          If there is none, the whole list will specify the number of output channels of the convolution layers
          and how to group the convolutions. Note that there is no need to specify the number of convolution layers.
          e.g.
            >>> channel_list = [(C_conv1, G_conv1), (C_conv2, G_conv2), ...]
          Be careful though, there may be some restrictions on the values (for graph_conv for example).
        sequence_length (int): Initial sequences temporal duration (number of time steps)
        activation_fct (str): The activation function to use for all modules. Can be 'relu', 'prelu' or 'swish'
        dropout (float): Dropout parameter between 0 and 1
        fc_hidden_layers (list): Size of the (not necessary) fully connected hidden layers of the classification module
        nb_classes (int): Number of output classes
    """

    def __init__(self, preprocess, conv_type, channel_list, fc_hidden_layers=[1936, 128],
                 sequence_length=100, activation_fct='prelu', dropout=0.4, nb_classes=14):

        # Init
        super(Net, self).__init__()
        activations = {
            'relu': nn.ReLU,
            'prelu': nn.PReLU,
            'swish': Swish,
        }
        self.activation_fct = activations[activation_fct]
        self.pool = nn.AvgPool1d(kernel_size=2)

        # Parameters quick check
        if preprocess is not None:
            input_channels = channel_list[0][0]  # int
            channel_list = channel_list[1:]  # tuple
            if len(channel_list) == 0:
                raise AttributeError('Channel list provided does not provide output dims for convolution layers')
        else:
            input_channels = 66
        output_channel_list = [tup[0] for tup in channel_list]
        groups = [tup[1] for tup in channel_list]
        if not (isinstance(groups, list) or isinstance(groups, tuple)):
            raise AttributeError('The ``channel_list`` argument expects a list of tuples.\n'
                                 'Example: channel_list = [<(C_preprocess, None)>, (C_conv1, G_conv1),'
                                 ' (C_conv2, G_conv2), (C_conv3, G_conv3), ...]')

        # Preprocess module
        if preprocess is 'LSC':
            self.preprocess_module = nn.Linear(66, output_channel_list[0])
        elif preprocess is 'graph_conv':
            self.preprocess_module = GCN(output_channel_list[0])
        elif preprocess is None:
            self.preprocess_module = None
        else:
            raise AttributeError('Preprocess module {} not recognized'.format(preprocess))

        # Convolution module
        if conv_type is 'regular':
            self.conv_small = RegularConvNet(input_channels, output_channel_list, groups, kernel_size=3,
                                             activation_fct=self.activation_fct, pool=self.pool, dropout=dropout)
            self.conv_large = RegularConvNet(input_channels, output_channel_list, groups, kernel_size=7,
                                             activation_fct=self.activation_fct, pool=self.pool, dropout=dropout)
        elif conv_type is 'temporal':
            self.conv_small = TemporalConvNet(input_channels, output_channel_list, groups, kernel_size=3,
                                              activation_fct=self.activation_fct, dropout=dropout)
            self.conv_large = TemporalConvNet(input_channels, output_channel_list, groups, kernel_size=7,
                                              activation_fct=self.activation_fct, dropout=dropout)
        else:
            raise AttributeError('Convolution module {} not recognized'.format(conv_type))

        # Classification module
        nb_features_1 = self.conv_small.get_out_features(sequence_length) + self.conv_large.get_out_features(
            sequence_length)
        self.fc_module = FullyConnected(input_layer=nb_features_1, hidden_layers=fc_hidden_layers,
                                        nb_classes=nb_classes, activation_fct_class=self.activation_fct,
                                        dropout_p=dropout)

    def forward(self, x):

        # Preprocess module
        if self.preprocess_module is not None:
            x = self.preprocess_module(x)

        # Convolution module
        x = x.transpose(1, 2)
        x_small = self.conv_small(x)
        x_large = self.conv_large(x)
        x = torch.cat([x_small, x_large], dim=1)

        # Classification module
        x = x.view(-1, training_utils.num_flat_features(x))
        x = self.fc_module(x)

        return x
