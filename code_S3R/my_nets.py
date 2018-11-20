import torch
from torch import nn

from code_S3R.modules.fully_connected import FullyConnected
from code_S3R.modules.regular_cn import RegularConvNet
from code_S3R.modules.tcn import TemporalConvNet
from code_S3R.my_utils import training_utils
from code_S3R.modules.gcn import GCN


# class Net(nn.Module):
#     """
#     [Devineau et al., 2018] Deep Learning for Hand Gesture Recognition on Skeletal Data
#
#     This model computes a succession of *n x[convolutions and pooling pipeline]* independently on different
#     sequence channels.
#
#     Each of these computations are actually done at two different resolutions, that are later merged by concatenation
#     with the (pooled) original sequence channels.
#
#     In short:
#
#     1. input (N, C_in=n*m, L_in) --> split into n separate pipelines, each pipeline working on m sequence channels.
#
#     2. Convolutions
#         - pipeline_i --> [conv/pool/dropout] high_resolution_i
#
#         - pipeline_i --> [conv/pool/dropout] low_resolution_i
#
#         - pipeline_i --> pooled_i
#
#         - low_resolution_i, high_resolution_i, pooled_i --> output_i
#
#     3. [n x output_i] ---> output (N, C_out, L_out)
#     """
#
#     def __init__(self, activation_fct='prelu', dropout=0.4, net_type='xyz', net_shape=(0, 0), nb_classes=14):
#         r"""
#         Instantiates the parameters and the modules.
#
#         :param activation_fct: Activation function used (relu, prelu or swish).
#         :param dropout: Dropout parameter
#         :param net_type: an instance of NetType
#         :param net_shape:
#         For `linear_combination` networks : (nb_seq_per_pipeline, nb_pipelines)
#         For `graph_conv` networks : (nb_features_per_node, 22)
#         Otherwise it's already determined by the network type.
#         :param nb_classes: Number of output classes (gestures).
#         """
#         super(Net, self).__init__()
#         self.activation_fct = activation_fct
#         self.dropout = dropout
#         self.net_type = net_type
#         self.net_shape = net_shape
#
#
#         # m : number of sequence channels per pipeline
#         # n : number of pipelines
#         self.m, self.n = training_utils.get_network_shape(self.net_type, self.net_shape)
#
#         activations = {
#             'relu': nn.ReLU,
#             'prelu': nn.PReLU,
#             'swish': training_utils.Swish,
#         }
#         pool = nn.AvgPool1d(kernel_size=2)
#
#         # Eventually add a linear/graph_conv layer to mix the temporal sequences
#         self.preprocess_module = training_utils.get_preprocess_module(net_type, net_shape)
#
#         # High resolution branch
#         # The first module is shared by all channels
#         self.conv_high_res_shared = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv1d(in_channels=self.m, out_channels=8, kernel_size=7, padding=3),
#                 activations[activation_fct](),
#                 pool,
#                 nn.Dropout(self.dropout),
#             )
#         ])
#         # Each channel has a specific 2nd and 3rd module
#         self.conv_high_res_individuals = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv1d(in_channels=8, out_channels=4, kernel_size=7, padding=3),
#                 activations[activation_fct](),
#                 pool,
#                 nn.Dropout(self.dropout),
#
#                 nn.Conv1d(in_channels=4, out_channels=4, kernel_size=7, padding=3),
#                 activations[activation_fct](),
#                 pool,
#                 nn.Dropout(self.dropout),
#             )
#             for i in range(self.n)
#         ])
#
#         # Low resolution branch
#         # The first module is shared by all channels
#         self.conv_low_res_shared = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv1d(in_channels=self.m, out_channels=8, kernel_size=3, padding=1),
#                 activations[activation_fct](),
#                 pool,
#                 nn.Dropout(self.dropout),
#             )
#         ])
#         # Each channel has a specific 2nd and 3rd module
#         self.conv_low_res_individuals = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv1d(in_channels=8, out_channels=4, kernel_size=3, padding=1),
#                 activations[activation_fct](),
#                 pool,
#                 nn.Dropout(self.dropout),
#
#                 nn.Conv1d(in_channels=4, out_channels=4, kernel_size=3, padding=1),
#                 activations[activation_fct](),
#                 pool,
#                 nn.Dropout(self.dropout),
#             )
#             for i in range(self.n)
#         ])
#
#         # Residual branch, one for each channel
#         self.residual_modules = nn.ModuleList([
#             nn.Sequential(
#                 pool,
#                 pool,
#                 pool,
#             )
#             for i in range(self.n)
#         ])
#
#         # The last layer, fully connected
#         nb_features_1 = self.n * (
#                 4 + 4 + self.m) * 12  # Size : n pipelines * (4 + 4 + m) outputs/channel * 12 time step/output
#         # TODO : check other values ?
#         nb_features_2 = int(nb_features_1 / 3.91)
#         self.fc_module = nn.Sequential(
#             nn.Linear(in_features=nb_features_1, out_features=nb_features_2),
#             activations[activation_fct](),
#             nn.Dropout(self.dropout),
#
#             nn.Linear(in_features=nb_features_2, out_features=nb_classes),
#         )
#
#         # Xavier initialization
#         training_utils.perform_xavier_init(
#             [
#                 self.conv_high_res_shared,
#                 self.conv_high_res_individuals,
#                 self.conv_low_res_shared,
#                 self.conv_low_res_individuals,
#                 self.residual_modules
#             ],
#             [
#                 self.fc_module
#             ],
#             activation_fct
#         )
#
#     def forward(self, input_batch):
#         """
#         This function performs the actual computations of the network for a forward pass.
#
#         :param input_batch: A tensor of gestures. Its shape is (batch_size x temporal_duration x nb_sequences)
#         :return: A tensor of results. Its shape is (batch_size x nb_classes)
#         """
#         # Pre-process input if required
#         if self.net_type is NetType.graph_conv.value:
#             pipeline_inputs = self.preprocess_module(input_batch, adj=training_utils.adj)
#         else:
#             if self.preprocess_module is not None:
#                 input_batch = self.preprocess_module(input_batch)
#
#             # Apply pipelines
#             pipeline_inputs = training_utils.get_pipeline_inputs(input_batch=input_batch,
#                                                                  net_type=self.net_type, net_shape=self.net_shape)
#
#         pipeline_outputs = []
#
#         for i in range(self.n):
#             pipeline_input = pipeline_inputs[i]
#
#             # Apply the 3 branches
#             high = self.conv_high_res_shared[0](pipeline_input)
#             high = self.conv_high_res_individuals[i](high)
#
#             low = self.conv_low_res_shared[0](pipeline_input)
#             low = self.conv_low_res_individuals[i](low)
#
#             original = self.residual_modules[i](pipeline_input)
#
#             # Concatenate the results to a single channel output (along the channel axis, not the batch one)
#             channel_output = torch.cat([
#                 original,
#                 low,
#                 high
#             ], 1)
#             pipeline_outputs.append(channel_output)
#
#         # Concatenates all pipelines output to a single dimension output.
#         output = torch.cat(pipeline_outputs, 1)
#         output = output.view(-1, training_utils.num_flat_features(output))
#
#         output = self.fc_module(output)
#
#         return output
#
#
# class TCN(nn.Module):
#
#     def __init__(self, activation_fct='prelu', dropout=0.4, net_type='TCN', net_shape=(0, 0), tcn_channels=None,
#                  tcn_k=2,
#                  nb_classes=14):
#         super(TCN, self).__init__()
#         self.activation_fct = activation_fct
#         self.dropout = dropout
#         self.net_type = net_type
#         self.net_shape = net_shape
#         self.tcn_channels = tcn_channels
#         self.tcn_k = tcn_k
#
#         activations = {
#             'relu': nn.ReLU,
#             'prelu': nn.PReLU,
#             'swish': training_utils.Swish,
#         }
#         self.activation_fct = activations[activation_fct]
#
#         if tcn_channels is None:
#             tcn_channels = []
#
#         self.tcn = tcn.TemporalConvNet(66, num_channels=tcn_channels, kernel_size=tcn_k, dropout=dropout,
#                                        activation_fct=self.activation_fct)
#
#         nb_features_1 = tcn_channels[-1] * 100
#         nb_features_2 = int(nb_features_1 / 4)
#         self.fc_module = nn.Sequential(
#             nn.Linear(in_features=nb_features_1, out_features=nb_features_2),
#             activations[activation_fct](),
#             nn.Dropout(self.dropout),
#             nn.Linear(in_features=nb_features_2, out_features=nb_classes),
#         )
#
#     def forward(self, input_batch):
#         # reshape input_batch from (N, L, C) to (N, C, L)
#         input = input_batch.transpose(1, 2)
#
#         output = self.tcn(input)
#
#         output = output.view(-1, training_utils.num_flat_features(output))
#         output = self.fc_module(output)
#
#         return output


class Net2(nn.Module):

    def __init__(self, preprocess, conv_type, channel_list, sequence_length, groups=1, activation_fct='prelu',
                 dropout=0.4, fc_hidden_layers=[1936, 128], nb_classes=14):
        super(Net2, self).__init__()

        activations = {
            'relu': nn.ReLU,
            'prelu': nn.PReLU,
            'swish': training_utils.Swish,
        }
        self.activation_fct = activations[activation_fct]
        self.pool = nn.AvgPool1d(kernel_size=2)

        # Preprocess module
        if preprocess is 'LSC':
            self.preprocess_module = nn.Linear(66, channel_list[0])
        elif preprocess is 'graph_conv':
            self.preprocess_module = GCN(channel_list[0])
        elif preprocess is None:
            self.preprocess_module = None
        else:
            raise AttributeError('Preprocess module {} not recognized'.format(preprocess))

        # Convolution module
        if preprocess is not None:
            input_channels = channel_list[0]
            channel_list = channel_list[1:]
        else:
            input_channels = 66

        if conv_type is 'regular':
            self.conv_small = RegularConvNet(input_channels, channel_list, groups, kernel_size=3,
                                             activation_fct=self.activation_fct, pool=self.pool, dropout=dropout)
            self.conv_large = RegularConvNet(input_channels, channel_list, groups, kernel_size=7,
                                             activation_fct=self.activation_fct, pool=self.pool, dropout=dropout)
        elif conv_type is 'temporal':
            self.conv_small = TemporalConvNet(input_channels, channel_list, groups, kernel_size=3,
                                              activation_fct=activation_fct, dropout=dropout)
            self.conv_large = TemporalConvNet(input_channels, channel_list, groups, kernel_size=7,
                                              activation_fct=activation_fct, dropout=dropout)
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
