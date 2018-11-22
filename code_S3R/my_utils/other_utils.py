import pathlib
from datetime import datetime

import numpy
import pandas
from comet_ml import Experiment
import torch
from scipy import ndimage, random
from sklearn.utils import shuffle
from skorch.callbacks import Callback
import warnings


# Hide warnings
def hide_scipy_zoom_warnings():
    warnings.filterwarnings('ignore', '.*output shape of zoom.*')


# Get hyper params
def get_channels_list(nb_configs, preprocessing):

    def _gcl(preprocessing):
        max_depth = 5

        hyper_params = []

        for depth in range(1, max_depth+1):
            for G in [1, 3, 11, 22, 66]:

                channel_list = []

                if preprocessing:
                    # add preprocessing layer
                    upper_bound_preprocess = int(2 * 66 / G)
                    # Choose a random number of channels between 1 and 2*66, no matter the G
                    c = G * random.randint(1, upper_bound_preprocess + 1)
                    channel_list.append((c, None))

                # add convolution layers
                upper_bound_conv = int(2 * 66 / G)
                for layer in range(depth):
                    # Choose a random number of channels between 1 and 4*66, no matter the G
                    c = G * random.randint(1, upper_bound_conv + 1)
                    channel_list.append((c, G))

                hyper_params.append(channel_list)

        return hyper_params

    hyper_params_list = []
    for i in range(nb_configs):
        hyper_params_list += _gcl(preprocessing=preprocessing)

    return hyper_params_list

# Logging -------------

class MyCallback(Callback):
    """
    Log data at each run (hyper-parameters and metrics), and save it in a CSV file and/or in a comet_ml experiment.


    Args:
        param_keys_to_log (list): The names of the hyper-parameters to log
        search_run_id (str): Unique identifier or the grid search
        log_to_csv (bool):
        log_to_comet_ml (bool):

    """

    # place here attributes that are not fed as arguments to the __init__ method
    experiment = None
    params_to_log = None

    def __init__(self, param_keys_to_log, search_run_id, log_to_csv=True, log_to_comet_ml=True) -> None:
        self.param_keys_to_log = param_keys_to_log
        self.search_run_id = search_run_id
        self.log_to_csv = log_to_csv
        self.log_to_comet_ml = log_to_comet_ml

    def on_train_begin(self, net, **kwargs):
        """
        If log_to_comet_ml=True, create a comet.ml experiment and log hyper-parameters specified in params_to_log.

        If log_to_csv=True, create a new csv

        Args:
            net:
            **kwargs:

        """
        params = net.get_params()
        self.params_to_log = {}
        for key in self.param_keys_to_log:
            try:
                self.params_to_log[key] = params[key]
            except KeyError:
                # in case params[key] is not found (for some grid searches it can be the case)
                self.params_to_log[key] = None

        if self.log_to_comet_ml:
            self.experiment = Experiment(api_key='Tz0dKZfqyBRMdGZe68FxU3wvZ', project_name='S3R')
            self.experiment.log_multiple_params(self.params_to_log)
            self.experiment.set_model_graph(net.__str__())

            # make it easier to regroup experiments by grid_search
            self.experiment.log_other('search_run_id', self.search_run_id)

    def on_epoch_end(self, net, **kwargs):
        """
        Log epoch metrics to comet.ml if required
        """
        data = net.history[-1]

        if self.log_to_comet_ml:
            self.experiment.log_multiple_metrics(
                dic=dict((key, data[key]) for key in [
                    'valid_acc',
                    'valid_loss',
                    'train_loss',
                ]),
                step=data['epoch']
            )

    def on_train_end(self, net, X=None, y=None, **kwargs):
        """
        Save the metrics in a csv file if required
        """
        if self.log_to_csv:
            valid_acc_data = net.history[:, 'valid_acc']
            valid_loss_data = net.history[:, 'valid_loss']
            train_loss_data = net.history[:, 'train_loss']

            run_results = pandas.DataFrame({
                'params': self.params_to_log,
                'data': {
                    'valid_acc': valid_acc_data,
                    'valid_loss': valid_loss_data,
                    'train_loss': train_loss_data,
                }
            })

            file_name = 'data_{:%H%M%S}.csv'.format(datetime.now())

            # create the directory if does not exist, without raising an error if it does already exist
            pathlib.Path('results/{}'.format(self.search_run_id)).mkdir(parents=True, exist_ok=True)

            # save the file
            run_results.to_csv(path_or_buf='results/{}/{}'.format(self.search_run_id, file_name))


def save_and_print_results(search_run_id, cv_results, grid_search_params):
    """
    Save results as a csv file.
    Print :
        - Best score with the corresponding params
        - Filename of the csv file
        - All the results just in case

    Args:
        grid_search_params:
        cv_results:
        search_run_id:
    """
    results = pandas.DataFrame(cv_results).sort_values('rank_test_score')

    # create the directory if does not exist, without raising an error if it does already exist
    pathlib.Path('results/{}'.format(search_run_id)).mkdir(parents=True, exist_ok=True)

    # select important columns to save
    columns = ['rank_test_score', 'mean_test_score', 'std_test_score']
    for key in get_param_keys(grid_search_params):
        columns.append('param_' + key)
    columns.append('mean_fit_time')

    # save important results
    results.to_csv(path_or_buf='results/{}/summary.csv'.format(search_run_id),
                   columns=columns)
    # save all search results, without excluding some columns
    results.to_csv(path_or_buf='results/{}/detailed.csv'.format(search_run_id))

    print('------')
    print('Results saved with search_run_id {}'.format(search_run_id))
    print('All results just in case :\n', cv_results)


# Loading and pre-processing -------------

def load_data(filepath='./data/data.numpy.npy'):
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
                                              ndimage.zoom(x_i.T[j], final_length / len(x_i), mode='reflect') for j in
                                              range(numpy.size(x_i, 1))
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
