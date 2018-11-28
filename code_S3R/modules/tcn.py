import torch.nn as nn
from torch.nn.utils import weight_norm

from code_S3R.modules.attention import DotAttention, GeneralSelfAttention, TransposeAxesOneAndTwo


class TemporalConvNet(nn.Module):
    """
    Implementation of TCN model proposed in https://arxiv.org/abs/1803.01271

    Shape
        - Input : :math:`(N, C_{in}, L)`, where :math:`C_{in}` is the the number of input channels
        - Output : :math:`(N, C_{out}, L)`, where :math:`C_{out}` is the the number of output channels

    Args:
        num_inputs (int): Number of input channels
        num_channels (list): List containing the number of output channels of each layer
        groups (list): List containing the 'groups' parameter for each layer. Useful if you want to share some
            convolutions but not all of them. Be careful to check that, for each layer, groups can divide both
            C_in and C_out.
        kernel_size (int):
        activation_fct: The activation function class, ready to be called
        dropout (float):
    """

    def __init__(self, num_inputs, num_channels, groups, kernel_size, activation_fct, dropout, temporal_attention):
        super(TemporalConvNet, self).__init__()
        self.num_channels = num_channels
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, activation_fct=activation_fct,
                                     dropout=dropout, groups=groups[i])]
            if temporal_attention == 'dot_attention':
                layers += [TransposeAxesOneAndTwo(), DotAttention(), TransposeAxesOneAndTwo()]
            elif temporal_attention == 'general_attention':
                layers += [TransposeAxesOneAndTwo(), GeneralSelfAttention(C=out_channels), TransposeAxesOneAndTwo()]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def get_out_features(self, sequence_length):
        """
        Returns: Number of out features (L * C_out)
        """
        return self.num_channels[-1] * sequence_length


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, groups, activation_fct, dropout=0.2):

        super(TemporalBlock, self).__init__()

        # Block layers
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size=kernel_size,
                                           stride=stride, padding=padding, dilation=dilation, groups=groups))
        self.chomp1 = Chomp1d(padding)
        self.act1 = activation_fct()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size=kernel_size,
                                           stride=stride, padding=padding, dilation=dilation, groups=groups))
        self.chomp2 = Chomp1d(padding)
        self.act2 = activation_fct()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.act1, self.dropout1,
                                 self.conv2, self.chomp2, self.act2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.act = activation_fct()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.act(out + res)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()
