from torch import nn

from code_S3R.my_utils import training_utils


class FullyConnected(nn.Module):
    """
    Apply 2 successive fully connected layers, for classification.

    Shape:
        - Input : *(nb_features_1)* The high level features extracted by the convolution network, concatenated
        - Output : *(nb_classes)* The classification tensor

    Args:
        input_layer (int): Input size
        hidden_layers (list): List of integers. Each i-th integer is the number of features after the i-th layer.
        nb_classes (int): Output size (i.e. number of classes)
        activation_fct_class: Activation function class, ready to be instantiated
        dropout_p (float): Dropout parameter between 0 and 1. Default: None.
    """

    def __init__(self, input_layer, hidden_layers, nb_classes, activation_fct_class, dropout_p=None):
        super(FullyConnected, self).__init__()
        self.activation_fct_class = activation_fct_class
        self.activation_fct_name = self.activation_fct_class.__name__
        self.activation_fct_name = self.activation_fct_name.lower()
        if self.activation_fct_name is 'leakyrelu':
            self.activation_fct_name = 'leaky_relu'
        self.dropout_p = dropout_p

        nb_neurons = [input_layer] + hidden_layers + [nb_classes]  # merges the three lists

        layers = []
        for idx in range(len(nb_neurons) -1):
            layers.append(nn.Linear(in_features=nb_neurons[idx], out_features=nb_neurons[idx+1]))
            layers.append(self.activation_fct_class())
            if self.dropout_p is not None:
                layers.append(nn.Dropout(self.dropout_p))

        self.network = nn.Sequential(*layers)
        self.init_weights()

    def init_weights(self):
        training_utils.perform_xavier_init(module_lists=[], modules=[self.network],
                                           activation_fct=self.activation_fct_name)

    def forward(self, x):
        return self.network(x)
