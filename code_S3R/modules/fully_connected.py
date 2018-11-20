from torch import nn

from code_S3R.my_utils import training_utils


class FullyConnected(nn.Module):
    """
    Apply 2 successive fully connected layers, for classification.

    Shape:
        - Input : *(nb_features_1)* The high level features extracted by the convolution network, concatenated
        - Output : *(nb_classes)* The classification tensor

    Args:
        nb_features_1 (int): Number of input features
        nb_features_2 (int): Number of features between the 2 FC layers
        nb_classes (int): Number of classes as output
        activation_fct: Activation function class, ready to be instantiated
        activation_fct_name (string): The string corresponding to the activation_fct ('relu', 'prelu', 'swish', etc)
    """

    def __init__(self, nb_features_1, nb_features_2, nb_classes, activation_fct, activation_fct_name):
        super(FullyConnected, self).__init__()
        self.activation_fct_name = activation_fct_name

        self.fc_layer1 = nn.Linear(in_features=nb_features_1, out_features=nb_features_2),
        self.fc_layer2 = nn.Linear(in_features=nb_features_2, out_features=nb_classes),

        self.network = nn.Sequential(
            self.fc_layer1,
            activation_fct(),
            nn.Dropout(self.dropout),
            self.fc_layer2,
            # todo: add softmax ?
        )

    def init_weights(self):
        training_utils.perform_xavier_init(module_lists=[], modules=[self.network],
                                           activation_fct=self.activation_fct_name)

    def forward(self, x):
        return self.network(x)
