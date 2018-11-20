from torch import nn
from torch.nn.utils import weight_norm

from code_S3R.my_utils.training_utils import perform_xavier_init


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
            C_in and C_out. todo: really implement groups as a list
        kernel_size (int):
        activation_fct: The activation function class, ready to be called
        pool: The already instantiated pooling module
        dropout (float):
    """

    def __init__(self, num_inputs, num_channels, groups, kernel_size, activation_fct, pool, dropout):
        super(RegularConvNet, self).__init__()
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.pool = pool

        layers = []
        self.num_levels = len(num_channels)

        for i in range(self.num_levels):
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            padding = int((kernel_size - 1) / 2)
            layers += [RegularConvBlock(groups, in_channels, out_channels, kernel_size, padding, activation_fct, pool,
                                        dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def get_out_features(self, sequence_length):
        """

        Returns: Number of out features (L_out * C_out)
            L_out depends on the the convolutions and pooling applied :
                - For convolutions, with default stride=1, padding=int(kernel_size-1) and default dilatation=1 :
                  L_out = L_in if kernel_size is odd, L_out = L_in - 1 if kernel_size is even
                - For pooling, with default stride=pool_kernel_size and padding=0 :
                  L_out = int(L_in / pool_kernel_size)

        """
        pool_k = self.pool.kernel_size[0]  # kernel_size is a tuple
        conv_k = self.kernel_size

        length_out = sequence_length
        for i in range(self.num_levels):
            # Each level, one convolution and then one pooling is applied
            if conv_k % 2 == 0:
                length_out -= 1
            length_out = int(length_out / pool_k)

        return self.num_channels[-1] * length_out


class RegularConvBlock(nn.Module):
    """
    A module of 1 classical convolutions and a residual branch. Convolution may be applied on on sequences separately,
    by specifying ``groups``.

    Shape
        - Input : :math:`(N, C_{in}, L_{in})`, where :math:`C_{in}` is the the number of input channels
        - Output : :math:`(N, C_{out}, L_{out})`, where :math:`C_{out}` is the the number of output channels and
          :math:`L_{out}` the length of the sequences after applying pooling twice.

    """

    def __init__(self, groups, in_channels, out_channels, kernel_size, padding, activation_fct, pool, dropout):
        super(RegularConvBlock, self).__init__()
        self.activation_fct = activation_fct

        # Convolution branch
        self.conv = weight_norm(nn.Conv1d(in_channels, out_channels,
                                          kernel_size=kernel_size, padding=padding, groups=groups))

        self.net = nn.Sequential(
            self.conv,
            activation_fct(),
            pool,
            nn.Dropout(dropout),
        )

        # Residual branch
        if kernel_size % 2 == 1:
            residual_pool = pool
        else:
            k_pool_original = pool.kernel_size[0]  # kernel_size is a tuple
            residual_pool = nn.AvgPool1d(kernel_size=k_pool_original + 1,
                                         stride=k_pool_original)
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1),  # doesn't change sequence length
                residual_pool,
            )
        else:
            self.downsample = residual_pool
        self.act = activation_fct()

    def init_weights(self):
        perform_xavier_init(modules=[self.net, self.downsample],
                            activation_fct=self.activation_fct.__name__)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.act(out + res)
