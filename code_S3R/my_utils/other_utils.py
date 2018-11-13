from datetime import datetime

import numpy
import pandas
import torch
from comet_ml import Experiment
from scipy import ndimage
from sklearn.utils import shuffle
from skorch.callbacks import Callback


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
    results.to_csv(path_or_buf='./run-data/grid_searches/{}.csv'.format(file_name),
                   columns=columns)
    # save all results, without excluding some columns
    results.to_csv(path_or_buf='./run-data/grid_searches/detailed_grid_results/{}_all.csv'.format(file_name))

    print('------')
    print('Results saved as {}.csv'.format(file_name))
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
