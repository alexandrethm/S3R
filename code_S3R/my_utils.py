import itertools
import time
import torch
from scipy import ndimage

import numpy
import math

from sklearn.utils import shuffle
from torch import nn
from code_S3R import my_nets


# -------------
# Training
# -------------


def perform_training(x_train, y_train, x_test, y_test, hyper_params, experiment):
    # -------------
    # Network instantiation
    # -------------
    net = my_nets.XYZNet(activation_fct=hyper_params['activation_fct'])

    # -----------------------------------------------------
    # Loss function & Optimizer
    # -----------------------------------------------------
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=net.parameters(), lr=hyper_params['learning_rate'])

    # -------------
    # Training
    # -------------
    # Prepare all mini-batches
    x_train_batches = batch(
        x_train,
        batch_size=hyper_params['batch_size']
    )  # list of tensors (batch_size, temporal_duration, nb_sequences=66)
    y_train_batches = batch(
        y_train,
        batch_size=hyper_params['batch_size']
    )  # list of output tensors containing the index of the right gesture, size : (batch_size)

    for epoch in range(hyper_params['num_epochs']):
        current_loss = 0.0

        for i, (x_train_batch, y_train_batch) in enumerate(zip(x_train_batches, y_train_batches)):
            # zero the gradient parameters
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(
                x_train_batch)  # outputs is a tensor (batch_size, nb_classes=14) with the proba of each gesture
            loss = criterion(outputs, y_train_batch)
            loss.backward()
            optimizer.step()

            # for an easy access
            current_loss += loss.item()

        # Log to Comet.ml
        accuracy_test = get_accuracy(net, x_test, y_test)
        accuracy_train = get_accuracy(net, x_train, y_train)
        experiment.log_multiple_metrics({
            'accuracy_test': accuracy_test,
            'accuracy_train': accuracy_train,
            'loss': current_loss,
        }, step=epoch)

        print('Epoch [{}/{}] | Loss : {:.4e} | Accuracy_train : {:.4e} | Accuracy_test : {:.4e}'
              .format(epoch + 1, hyper_params['num_epochs'], current_loss, accuracy_train, accuracy_test))

    # -------
    # Requires a real test set, distinct from the training set (used for training the network)
    # and the validation set (used to find the parameters)
    # -------
    # with experiment.test():
    #     # x_test.size : (nb_test_gestures, temporal_duration, nb_sequences=66)
    #     # y_test.size : (nb_test_gestures), containing indexes of the right gestures
    #     outputs = net(x_test)  # outputs.size : (nb_test_gestures, nb_classes=14), containing probas of each gesture
    #     _, predicted = torch.max(outputs.data, 1)
    #
    #     total = y_test.size(0)
    #     correct = (predicted == y_test).sum()  # correct.size : (nb_test_gestures), containing 1 if
    # prediction was right
    #     correct = correct.item()
    #
    #     accuracy = 100 * correct / total
    #     experiment.log_metric('accuracy', accuracy)
    #     print('Test Accuracy of the model on the {} test images: {}%'.format(y_test.size(0), accuracy))


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


# -------------
# Loading and pre-processing
# -------------

def load_data(filepath='/Users/alexandre/development/S3R/data/data.numpy.npy'):
    """
    Returns hand gesture sequences (X) and their associated labels (Y).
    Each sequence has two different labels.
    The first label Y describes the gesture class out of 14 possible gestures (e.g. swiping your hand to the right).
    The second label Y describes the gesture class out of 28 possible gestures (e.g. swiping your hand to the right with your index pointed, or not pointed).
    """
    data = numpy.load(filepath, encoding='latin1')
    # data = [x_train, x_test, y_train_14, y_train_28, y_test_14, y_test_28]
    return data[0], data[1], data[2], data[3], data[4], data[5]


def resize_sequences_length(x_train, x_test, final_length=100):
    """
    Resize the time series by interpolating them to the same length
    """
    x_train = numpy.array([
        numpy.array([
            ndimage.zoom(x_i.T[j], final_length / len(x_i), mode='reflect') for j in range(numpy.size(x_i, 1))
        ]).T
        for x_i
        in x_train
    ])
    x_test = numpy.array([
        numpy.array([
            ndimage.zoom(x_i.T[j], final_length / len(x_i), mode='reflect') for j in
            range(numpy.size(x_i, 1))
        ]).T
        for x_i
        in x_test
    ])
    return x_train, x_test


def shuffle_dataset(x_train, x_test, y_train_14, y_train_28, y_test_14, y_test_28):
    """Shuffle the train/test data consistently."""
    # note: add random_state=0 for reproducibility
    x_train, y_train_14, y_train_28 = shuffle(x_train, y_train_14, y_train_28)
    x_test, y_test_14, y_test_28 = shuffle(x_test, y_test_14, y_test_28)
    return x_train, x_test, y_train_14, y_train_28, y_test_14, y_test_28


def preprocess_data(x_train, x_test, y_train_14, y_train_28, y_test_14, y_test_28, temporal_duration=100):
    """
    Preprocess the data as you want.
    """
    x_train, x_test, y_train_14, y_train_28, y_test_14, y_test_28 = shuffle_dataset(x_train, x_test, y_train_14,
                                                                                    y_train_28, y_test_14, y_test_28)
    x_train, x_test = resize_sequences_length(x_train, x_test, final_length=temporal_duration)

    return x_train, x_test, y_train_14, y_train_28, y_test_14, y_test_28


def convert_to_pytorch_tensors(x_train, x_test, y_train_14, y_train_28, y_test_14, y_test_28):
    y_train_14 = numpy.array(y_train_14)
    y_train_28 = numpy.array(y_train_28)
    y_test_14 = numpy.array(y_test_14)
    y_test_28 = numpy.array(y_test_28)

    # Remove 1 to all classes items (1-14 => 0-13 and 1-28 => 0-27)
    y_train_14 = y_train_14 - 1
    y_train_28 = y_train_28 - 1
    y_test_14 = y_test_14 - 1
    y_test_28 = y_test_28 - 1

    x_train = torch.from_numpy(x_train)
    x_test = torch.from_numpy(x_test)
    y_train_14 = torch.from_numpy(y_train_14)
    y_train_28 = torch.from_numpy(y_train_28)
    y_test_14 = torch.from_numpy(y_test_14)
    y_test_28 = torch.from_numpy(y_test_28)

    x_train = x_train.type(torch.FloatTensor)
    x_test = x_test.type(torch.FloatTensor)
    y_train_14 = y_train_14.type(torch.LongTensor)
    y_test_14 = y_test_14.type(torch.LongTensor)
    y_train_28 = y_train_28.type(torch.LongTensor)
    y_test_28 = y_test_28.type(torch.LongTensor)

    return x_train, x_test, y_train_14, y_train_28, y_test_14, y_test_28


# -------------
# Misc.
# -------------

def batch(tensor, batch_size=32):
    """Return a list of (mini) batches"""
    tensor_list = []
    length = tensor.shape[0]
    i = 0
    while True:
        if (i + 1) * batch_size >= length:
            tensor_list.append(tensor[i * batch_size: length])
            return tensor_list
        tensor_list.append(tensor[i * batch_size: (i + 1) * batch_size])
        i += 1


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '{:02d}m {:02d}s'.format(int(m), int(s))


def get_accuracy(model, x_tensor, y_tensor_real):
    """
    Params:
        :param model: a **pytorch** model
        :param x_tensor:  a **pytorch** tensor/Variable
        :param y_tensor_real:  a **pytorch** tensor/Variable
    """
    with torch.no_grad():
        model.eval()
        predicted = model(x_tensor)
        predicted = predicted.max(dim=1)[1]
        s = (predicted.long() == y_tensor_real.long()).sum().item()
        s = 1.0 * s / y_tensor_real.size()[0]
        model.train()
        return s


def xavier_init(layer, activation_fct):
    param = None

    if activation_fct == 'prelu':
        activation_fct = 'leaky_relu'
        param = torch.nn.PReLU().weight.item()
    elif activation_fct == 'swish':
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


def xavier_init_module(module_lists, modules, activation_fct):
    """
    Perform xavier_init on Conv1d and Linear layers insides the specified modules.
    :param module_lists: list of ModuleList objects
    :param modules: list of Module objects
    :param activation_fct:
    """
    for module in itertools.chain(module_lists):
        for layer in module:
            if layer.__class__.__name__ == "Conv1d" or layer.__class__.__name__ == "Linear":
                xavier_init(layer, activation_fct)

    for layer in itertools.chain(modules):
        if layer.__class__.__name__ == "Conv1d" or layer.__class__.__name__ == "Linear":
            xavier_init(layer, activation_fct)
