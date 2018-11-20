from torch import nn
from torch.nn.utils import weight_norm


class RegularConvNet(nn.Module):
    """
    Apply multiple RegularConvBlock, at a certain resolution (*kernel_size*).

    Convolutions may be applied separately on the channels, by specifying  *groups*.

    Shape
        - Input : :math:`(N, C_{in}, L_{in})`
        - Output : :math:`(N, C_{out}, L_{out})`

    Args:
        num_inputs (int): Number of input channels
        num_channels (list): List containing the number of output channels of each layer
        groups (list): List containing the 'groups' parameter for each layer. Useful if you want to share some
            convolutions but not all of them. Be careful to check that, for each layer, groups can divide both
            C_in and C_out.
        kernel_size (int):
        activation_fct: The activation function class, ready to be called
        pool: The already instantiated pooling module
        dropout (float):
    """

    def __init__(self, num_inputs, num_channels, groups, kernel_size, activation_fct, pool, dropout):
        super(RegularConvNet, self).__init__()

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            padding = int((kernel_size - 1) / 2)
            layers += [RegularConvBlock(groups, in_channels, out_channels, kernel_size, padding, activation_fct, pool,
                                        dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class RegularConvBlock(nn.Module):
    """
    A module of 2 classical convolutions and a residual branch. Convolutions may be applied on on sequences separately,
    by specifying *groups*.

    Shape
        - Input : :math:`(N, C_{in}, L_{in})`, where :math:`C_{in}` is the the number of input channels
        - Output : :math:`(N, C_{out}, L_{out})`, where :math:`C_{out}` is the the number of output channels and
        :math:`L_{out}` the length of the sequences after applying pooling twice.
    """

    def __init__(self, groups, in_channels, out_channels, kernel_size, padding, activation_fct, pool, dropout):
        super(RegularConvBlock, self).__init__()

        # Convolution branch
        # todo: only 1 conv layer per block
        conv1 = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size, padding, groups))
        conv2 = weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size, padding, groups))

        self.net = nn.Sequential(
            conv1,
            activation_fct(),
            pool,
            nn.Dropout(dropout),

            conv2,
            activation_fct(),
            pool,
            nn.Dropout(dropout),
        )

        # Residual branch
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def init_weights(self):
        # TODO : use xavier_init
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.act(out + res)
