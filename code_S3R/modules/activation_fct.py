import torch
from torch import nn


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
