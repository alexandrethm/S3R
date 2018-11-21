import torch


def xavier_init(layer, activation_fct):
    param = None

    if activation_fct == 'ReLU':
        activation_fct = 'relu'
    elif activation_fct == 'PReLU' or activation_fct == 'prelu':
        activation_fct = 'leaky_relu'
        param = torch.nn.PReLU().weight.item()
    elif activation_fct == 'Swish':
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


def perform_xavier_init(module_list, activation_fct):
    """
    Perform xavier_init on Conv1d and Linear layers insides the specified modules.

    Args:
        module_list (list): list of Module objects (Conv1d, Linear, or Sequence containing these modules)
        activation_fct (str): The class of the activation function

    Returns:

    """
    for module in module_list:
        if module.__class__.__name__ == 'Conv1d' or module.__class__.__name__ == 'Linear':
            xavier_init(module, activation_fct)
        elif module.__class__.__name__ == 'Sequential':
            for sub_module in module:
                if sub_module.__class__.__name__ == 'Conv1d' or sub_module.__class__.__name__ == 'Linear':
                    xavier_init(sub_module, activation_fct)
