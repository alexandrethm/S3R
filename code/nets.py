import itertools
import torch
from torch import nn


class RegularNet(nn.Module):
    """
    [Devineau et al., 2018] Deep Learning for Hand Gesture Recognition on Skeletal Data

    This model computes a succession of 3x [convolutions and pooling] independently on each of the 66 sequence channels.

    Each of these computations are actually done at two different resolutions, that are later merged by concatenation
    with the (pooled) original sequence channel.

    Finally, a multi-layer perceptron merges all of the processed channels and outputs a classification.

    In short:

        1. input --> split into 66 channels

        2.1. channel_i --> 3x [conv/pool/dropout] low_resolution_i
        2.2. channel_i --> 3x [conv/pool/dropout] high_resolution_i
        2.3. channel_i --> pooled_i

        2.4. low_resolution_i, high_resolution_i, pooled_i --> output_channel_i

        3. MLP(66x [output_channel_i]) ---> classification


    Note: "joint" is a synonym of "channel".

    """

    def __init__(self, nb_channels=66, nb_classes=14):
        """
        Instantiates the parameters and the modules.
        :param nb_channels: Number of time sequences (channels) in a data sequence.
        :param nb_classes: Number of output classes (gestures).
        """
        super(RegularNet, self).__init__()

        # High resolution branch
        # The first module is shared by all channels
        self.conv_high_res_shared = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=8, kernel_size=7, padding=3),
                nn.ReLU(),
                nn.AvgPool1d(kernel_size=2),
            )
        ])
        # Each channel has a specific 2nd and 3rd module
        self.conv_high_res_individuals = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=8, out_channels=4, kernel_size=7, padding=3),
                nn.ReLU(),
                nn.AvgPool1d(kernel_size=2),

                nn.Conv1d(in_channels=4, out_channels=4, kernel_size=7, padding=3),
                nn.ReLU(),
                nn.AvgPool1d(kernel_size=2),
            )
            for i in range(nb_channels)
        ])

        # Low resolution branch
        # The first module is shared by all channels
        self.conv_low_res_shared = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AvgPool1d(kernel_size=2),
            )
        ])
        # Each channel has a specific 2nd and 3rd module
        self.conv_low_res_individuals = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=8, out_channels=4, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AvgPool1d(kernel_size=2),

                nn.Conv1d(in_channels=4, out_channels=4, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AvgPool1d(kernel_size=2),
            )
            for i in range(nb_channels)
        ])

        # Residual branch, one for each channel
        self.residual_modules = nn.ModuleList([
            nn.Sequential(
                nn.AvgPool1d(2),
                nn.AvgPool1d(2),
                nn.AvgPool1d(2),
            )
            for i in range(nb_channels)
        ])

        # The last layer, fully connected
        self.fc_module = nn.Sequential(
            nn.Linear(in_features=9 * 66 * 12, out_features=1936),
            nn.ReLU(),
            nn.Linear(in_features=1936, out_features=nb_classes),
        )

        # Initialization --------------------------------------
        # Xavier init
        for module in itertools.chain(self.conv_high_res_shared, self.conv_high_res_individuals,
                                      self.conv_low_res_shared, self.conv_low_res_individuals,
                                      self.residual_modules):
            for layer in module:
                if layer.__class__.__name__ == "Conv1d":
                    nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
                    nn.init.constant_(layer.bias, 0.1)

        for layer in self.fc_module:
            if layer.__class__.__name__ == "Linear":
                nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.constant_(layer.bias, 0.1)

    def forward(self, input):
        """
        This function performs the actual computations of the network for a forward pass.

        :param input: A tensor of gestures. Its shape is (batch_size x temporal_duration x nb_channels)
        :return: A tensor of results. Its shape is (batch_size x nb_classes)
        """

        _, _, nb_channels = input.size()
        channel_output_list = []

        for i in range(nb_channels):
            channel_input = input[:, :, i]
            # Add a dummy (spatial) dimension for the time convolutions
            # Conv1D format : (batch_size, num_feature_maps, length_of_seq)
            channel_input = channel_input.unsqueeze(1)

            # Apply the 3 branches
            high = self.conv_high_res_shared[0](channel_input)
            high = self.conv_high_res_individuals[i](high)

            low = self.conv_low_res_shared[0](channel_input)
            low = self.conv_low_res_individuals[i](low)

            original = self.residual_modules[i](channel_input)

            # Concatenate the results to a single channel output (along the channel axis, not the batch one)
            channel_output = torch.cat([
                original,
                low,
                high
            ], 1)
            channel_output_list.append(channel_output)

        # Concatenates all channels output to a single dimension output.
        # Size : 66 channels * (4 + 4 + 1) outputs/channel * 12 time step/output
        output = torch.cat(channel_output_list, 1)
        output = output.view(-1, 9 * 66 * 12)

        output = self.fc_module(output)

        return output


class XYZNet(nn.Module):
    """
    [Devineau et al., 2018] Deep Learning for Hand Gesture Recognition on Skeletal Data

    This model computes a succession of 3x [convolutions and pooling] independently on each of the 3 coordinates (x, y, z).

    Each of these computations are actually done at two different resolutions, that are later merged by concatenation
    with the (pooled) original sequence.

    Finally, a multi-layer perceptron merges all of the processed sequences and outputs a classification.

    In short:

        1. input --> split into 3 channels (one for x_i sequences, one for y_i sequences, one for z_i sequences)

        2.1. sequence_i --> 3x [conv/pool/dropout] low_resolution_i
        2.2. sequence_i --> 3x [conv/pool/dropout] high_resolution_i
        2.3. sequence_i --> pooled_i

        2.4. low_resolution_i, high_resolution_i, pooled_i --> output_sequence_i

        3. MLP(3x [output_channel_i]) ---> classification


    """

    def __init__(self, activation='relu', nb_sequences=66, nb_classes=14):
        """
        Instantiates the parameters and the modules.
        :param activation: Activation function used (relu, prelu or swish).
        :param nb_sequences: Number of time sequences in a data sequence.
        :param nb_classes: Number of output classes (gestures).
        """
        super(XYZNet, self).__init__()
        self.activation_fct = activation

        activations = {
            'relu': nn.ReLU,
            'prelu': nn.PReLU,
            'swish': Swish,
        }
        pool = nn.AvgPool1d(kernel_size=2)

        # High resolution branch
        # The first module is shared by all channels
        self.conv_high_res_shared = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=22, out_channels=8, kernel_size=7, padding=3),
                activations[activation](),
                pool,
            )
        ])
        # Each channel has a specific 2nd and 3rd module
        self.conv_high_res_individuals = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=8, out_channels=4, kernel_size=7, padding=3),
                activations[activation](),
                pool,

                nn.Conv1d(in_channels=4, out_channels=4, kernel_size=7, padding=3),
                activations[activation](),
                pool,
            )
            for i in range(3)
        ])

        # Low resolution branch
        # The first module is shared by all channels
        self.conv_low_res_shared = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=22, out_channels=8, kernel_size=3, padding=1),
                activations[activation](),
                pool,
            )
        ])
        # Each channel has a specific 2nd and 3rd module
        self.conv_low_res_individuals = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=8, out_channels=4, kernel_size=3, padding=1),
                activations[activation](),
                pool,

                nn.Conv1d(in_channels=4, out_channels=4, kernel_size=3, padding=1),
                activations[activation](),
                pool,
            )
            for i in range(3)
        ])

        # Residual branch, one for each channel
        self.residual_modules = nn.ModuleList([
            nn.Sequential(
                pool,
                pool,
                pool,
            )
            for i in range(3)
        ])

        # The last layer, fully connected
        self.fc_module = nn.Sequential(
            nn.Linear(in_features=30 * 3 * 12, out_features=276),
            activations[activation](),
            nn.Linear(in_features=276, out_features=nb_classes),
        )

        # Initialization --------------------------------------
        # Xavier init
        for module in itertools.chain(self.conv_high_res_shared, self.conv_high_res_individuals,
                                      self.conv_low_res_shared, self.conv_low_res_individuals,
                                      self.residual_modules):
            for layer in module:
                if layer.__class__.__name__ == "Conv1d":
                    nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
                    nn.init.constant_(layer.bias, 0.1)

        for layer in self.fc_module:
            if layer.__class__.__name__ == "Linear":
                nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.constant_(layer.bias, 0.1)

    def forward(self, input):
        """
        This function performs the actual computations of the network for a forward pass.

        :param input: A tensor of gestures. Its shape is (batch_size x temporal_duration x nb_sequences)
        :return: A tensor of results. Its shape is (batch_size x nb_classes)
        """

        _, _, nb_sequences = input.size()
        channel_output_list = []

        for i in range(3):
            if nb_sequences % 3 != 0:
                raise ArithmeticError
            else:
                nb_seq_per_channel = nb_sequences // 3

            # Get all x_i (or y_i or z_i) time sequences in a list,
            # each one with the following shape : (batch_size, 1, temporal_duration)
            channel_inputs = [input[:, :, 3 * j + i].unsqueeze(1) for j in range(nb_seq_per_channel)]

            # Concatenate the list to get an appropriate shape : (batch_size, nb_sequences/3, temporal_duration),
            # so it fits Conv1D format : (batch_size, num_feature_maps, length_of_seq)
            channel_input = torch.cat(channel_inputs, 1)

            # Apply the 3 branches
            high = self.conv_high_res_shared[0](channel_input)
            high = self.conv_high_res_individuals[i](high)

            low = self.conv_low_res_shared[0](channel_input)
            low = self.conv_low_res_individuals[i](low)

            original = self.residual_modules[i](channel_input)

            # Concatenate the results to a single channel output (along the channel axis, not the batch one)
            channel_output = torch.cat([
                original,
                low,
                high
            ], 1)
            channel_output_list.append(channel_output)

        # Concatenates all channels output to a single dimension output.
        # Size : 3 channels * (4 + 4 + 22) outputs/channel * 12 time step/output
        output = torch.cat(channel_output_list, 1)
        output = output.view(-1, 30 * 3 * 12)

        output = self.fc_module(output)

        return output


class Swish(nn.Module):
    r"""Applies element-wise the function
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
        >>> m = nn.PReLU()
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
