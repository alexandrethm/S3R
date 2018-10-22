import itertools
import torch
from torch import nn

from code_S3R import my_utils as my_utils


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

    def __init__(self, activation_fct='relu', dropout=0, nb_classes=14):
        """
        Instantiates the parameters and the modules.
        :param activation_fct: Activation function used (relu, prelu or swish).
        :param nb_classes: Number of output classes (gestures).
        """
        super(XYZNet, self).__init__()
        self.activation_fct = activation_fct
        self.dropout = 0

        activations = {
            'relu': nn.ReLU,
            'prelu': nn.PReLU,
            'swish': my_utils.Swish,
        }
        pool = nn.AvgPool1d(kernel_size=2)

        # High resolution branch
        # The first module is shared by all channels
        self.conv_high_res_shared = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=22, out_channels=8, kernel_size=7, padding=3),
                activations[activation_fct](),
                pool,
                nn.Dropout(self.dropout),
            )
        ])
        # Each channel has a specific 2nd and 3rd module
        self.conv_high_res_individuals = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=8, out_channels=4, kernel_size=7, padding=3),
                activations[activation_fct](),
                pool,
                nn.Dropout(self.dropout),

                nn.Conv1d(in_channels=4, out_channels=4, kernel_size=7, padding=3),
                activations[activation_fct](),
                pool,
                nn.Dropout(self.dropout),
            )
            for i in range(3)
        ])

        # Low resolution branch
        # The first module is shared by all channels
        self.conv_low_res_shared = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=22, out_channels=8, kernel_size=3, padding=1),
                activations[activation_fct](),
                pool,
                nn.Dropout(self.dropout),
            )
        ])
        # Each channel has a specific 2nd and 3rd module
        self.conv_low_res_individuals = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=8, out_channels=4, kernel_size=3, padding=1),
                activations[activation_fct](),
                pool,
                nn.Dropout(self.dropout),

                nn.Conv1d(in_channels=4, out_channels=4, kernel_size=3, padding=1),
                activations[activation_fct](),
                pool,
                nn.Dropout(self.dropout),
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
            activations[activation_fct](),
            nn.Dropout(self.dropout),

            nn.Linear(in_features=276, out_features=nb_classes),
        )

        # Xavier initialization
        my_utils.perform_xavier_init(
            [
                self.conv_high_res_shared,
                self.conv_high_res_individuals,
                self.conv_low_res_shared,
                self.conv_low_res_individuals,
                self.residual_modules
            ],
            [
                self.fc_module
            ],
            activation_fct
        )

    def forward(self, input_batch):
        """
        This function performs the actual computations of the network for a forward pass.

        :param input_batch: A tensor of gestures. Its shape is (batch_size x temporal_duration x nb_sequences)
        :return: A tensor of results. Its shape is (batch_size x nb_classes)
        """

        _, _, nb_sequences = input_batch.size()
        channel_output_list = []

        for i in range(3):
            if nb_sequences % 3 != 0:
                raise ArithmeticError
            else:
                nb_seq_per_channel = nb_sequences // 3

            # Get all x_i (or y_i or z_i) time sequences in a list,
            # each one with the following shape : (batch_size, 1, temporal_duration)
            channel_inputs = [input_batch[:, :, 3 * j + i].unsqueeze(1) for j in range(nb_seq_per_channel)]

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
        output = output.view(-1, my_utils.num_flat_features(output))

        output = self.fc_module(output)

        return output
