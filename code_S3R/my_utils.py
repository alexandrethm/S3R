import itertools
from datetime import datetime
from enum import Enum

import pandas

import numpy
import torch
from comet_ml import Experiment
from scipy import ndimage
from sklearn.utils import shuffle
from skorch.callbacks import Callback
from torch import nn


# Logging -------------

class MyCallback(Callback):
    """
    Calls comet.ml methods to log data at each run.
    """

    experiment = None

    def __init__(self, params_to_log) -> None:
        self.params_to_log = params_to_log

    def on_train_begin(self, net, **kwargs):
        """
        Create a comet.ml experiment and log hyper-parameters specified in params_to_log.
        :param net:
        :return:
        """
        params = net.get_params()
        params_to_log = {}
        for key in self.params_to_log:
            try:
                params_to_log[key] = params[key]
            except KeyError:
                # in case params[key] is not found (for some grid searches it can be the case)
                params_to_log[key] = None

        self.experiment = Experiment(api_key='Tz0dKZfqyBRMdGZe68FxU3wvZ', project_name='S3R')
        self.experiment.log_multiple_params(params_to_log)
        self.experiment.set_model_graph(net.__str__())

    def on_epoch_end(self, net, **kwargs):
        """
        Log epoch metrics to comet.ml
        :param net:
        :param kwargs:
        :return:
        """
        data = net.history[-1]
        self.experiment.log_multiple_metrics(
            dic=dict((key, data[key]) for key in [
                'valid_acc',
                'valid_loss',
                'train_loss',
            ]),
            step=data['epoch']
        )


def save_and_print_results(cv_results, grid_search_params):
    """
    Save results as a csv file.
    Print :
        - Best score with the corresponding params
        - Filename of the csv file
        - All the results just in case
    :param cv_results: A GridSearchCV.cv_results_ attribute
    :param grid_search_params:
    :return:
    """
    results = pandas.DataFrame(cv_results).sort_values('rank_test_score')

    # select filename and important columns to save
    file_name = 'grid_{:%m%d_%H%M}'.format(datetime.now())
    columns = ['rank_test_score', 'mean_test_score', 'std_test_score']
    for key in get_param_keys(grid_search_params):
        columns.append('param_' + key)
    columns.append('mean_fit_time')

    # save important results
    results.to_csv(path_or_buf='../run-data/grid_searches/{}.csv'.format(file_name),
                   columns=columns)
    # save all results, without excluding some columns
    results.to_csv(path_or_buf='../run-data/grid_searches/detailed_grid_results/{}_all.csv'.format(file_name))

    print('------')
    print('Results saved as {}.csv'.format(file_name))
    print('All results just in case :\n', cv_results)


# Training -------------


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


class NetType(Enum):
    xyz = 'xyz'
    '''
    'xyz' for a XYZ network, with 3 pipelines (one for x_i sequences, one for y_i sequences, one for z_i sequences)
    '''

    regular = 'regular'
    '''
    'regular' for a network with 1 sequence per pipeline
    '''

    LSC = 'LSC'
    '''
    'LSC' (linear spatial combination) for a linear combination to be applied on the sequences before the conv/pool
    pipelines, in order to mix them.
    
    The same linear transformation is applied on all the sequences at each time step.
    
        input_batch : (batch_size, temporal_duration, initial_nb_sequences=66) ->
            preprocessed_batch : (batch_size, temporal_duration, nb_sequences=m*n)
            
    With m*n (nb_seq_per_pipeline*nb_pipelines) not necessarily the initial nb of sequences.
    '''


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


def perform_xavier_init(module_lists, modules, activation_fct):
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


def get_preprocess_module(net_type, net_shape):
    if net_type is NetType.LSC:
        m, n = net_shape
        return nn.Linear(66, m * n)
    else:
        return None


def get_network_shape(net_type, net_shape):
    """
    :param net_type: str
    :param net_shape: (nb_seq_per_pipeline, nb_pipelines) Only used for `linear_combination` networks, otherwise
    it's already determined by the network type.
    :return: m (number of of channels per pipeline), n (number of pipelines).
    """
    if net_type is NetType.xyz.value:
        return 22, 3
    elif net_type is NetType.regular.value:
        return 1, 66
    elif net_type is NetType.LSC.value:
        return net_shape


def get_pipeline_inputs(input_batch, net_type, net_shape):
    """
    Get inputs in the right shape, according to the network type specified.

    :param input_batch: A tensor of gestures. Its shape is (batch_size, temporal_duration, nb_sequences)
    :param net_type: str
    :param net_shape: Network shape (m, n) if required (for networks with pre-processing for instance)
    :return: A list containing the inputs for each pipeline. Its shape is [n x (batch_size, m, temporal_duration)]
    With each of the n pipelines fed with m sequences, it is NECESSARY to have m x n = nb_sequences, where nb_sequences
    is the total number of sequences provided as input.

    Note that we are talking about (possibly) preprocessed input and sequences, so nb_sequences is not necessarily
    the initial nb of sequences (66 for the SHREC data).
    """
    _, _, nb_sequences = input_batch.size()
    m, n = get_network_shape(net_type, net_shape)
    # make sure the dimensions are right
    assert m * n == nb_sequences, 'Number of sequences in batch input does not match network shape'

    pipeline_inputs = []

    if net_type is NetType.xyz.value:
        for i in range(n):
            # Get all x_i (or y_i or z_i) time sequences in a list,
            # each one with the following shape : (batch_size, 1, temporal_duration)
            pipeline_input = [input_batch[:, :, 3 * j + i].unsqueeze(1) for j in range(m)]

            # Concatenate the list to get an appropriate shape : (batch_size, m=22, temporal_duration),
            # so it fits Conv1D format : (batch_size, num_feature_maps, length_of_seq)
            pipeline_input = torch.cat(pipeline_input, 1)

            pipeline_inputs.append(pipeline_input)
        return pipeline_inputs

    elif net_type is NetType.regular.value:
        # Get all time sequences in a list,
        # each one with the following shape : (batch_size, m=1, temporal_duration)
        # so it fits Conv1D format : (batch_size, num_feature_maps, length_of_seq)
        pipeline_inputs = [input_batch[:, :, j].unsqueeze(1) for j in range(n)]

        return pipeline_inputs

    elif net_type is NetType.LSC.value:
        # Split the input batch into a list of n pipeline_input
        for i in range(n):
            # Add m time sequences to the input,
            # each one with the following shape : (batch_size, 1, temporal_duration)
            pipeline_input = [input_batch[:, :, i * m + j].unsqueeze(1) for j in range(m)]

            # Concatenate the list to get an appropriate shape : (batch_size, m, temporal_duration),
            # so it fits Conv1D format : (batch_size, num_feature_maps, length_of_seq)
            pipeline_input = torch.cat(pipeline_input, 1)

            pipeline_inputs.append(pipeline_input)
        return pipeline_inputs


# Loading and pre-processing -------------

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

    y_train_14 = numpy.array(y_train_14)
    y_train_28 = numpy.array(y_train_28)
    y_test_14 = numpy.array(y_test_14)
    y_test_28 = numpy.array(y_test_28)

    # Remove 1 to all classes items (1-14 => 0-13 and 1-28 => 0-27)
    y_train_14 = y_train_14 - 1
    y_train_28 = y_train_28 - 1
    y_test_14 = y_test_14 - 1
    y_test_28 = y_test_28 - 1

    return x_train, x_test, y_train_14, y_train_28, y_test_14, y_test_28


def convert_to_pytorch_tensors(x_train, x_test, y_train_14, y_train_28, y_test_14, y_test_28):
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


def get_param_keys(params):
    """
    :param params: dict of parameters, or list containing dicts of parameters
    :return: all keys contained inside the dict or inside the dicts of the list
    """
    if params.__class__ == dict:
        params_keys = params.keys()
    elif params.__class__ == list:
        params_keys = []
        for i in range(len(params)):
            for key in params[i]:
                if key not in params_keys:
                    params_keys.append(key)
    else:
        params_keys = []

    return list(params_keys)
