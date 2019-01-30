import os
import pathlib
from datetime import datetime

import pandas
import torch
from comet_ml import Experiment
from skorch.callbacks import Callback


# Network utils -------------
def xavier_init(layer, activation_fct):
    param = None

    if activation_fct == 'ReLU':
        activation_fct = 'relu'
    elif activation_fct == 'PReLU':
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


# Logging -------------
class S3RTrainingLoggerCallback(Callback):
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
            self.experiment = Experiment(api_key=os.environ['COMET_ML_API_KEY'], project_name='S3R-V3')
            self.experiment.log_parameters(self.params_to_log)
            self.experiment.set_model_graph(net.__str__())

            # make it easier to regroup experiments by grid_search
            self.experiment.log_other('search_run_id', self.search_run_id)

    def on_epoch_end(self, net, **kwargs):
        """
        Log epoch metrics to comet.ml if required
        """
        data = net.history[-1]

        if self.log_to_comet_ml:
            self.experiment.log_metrics(
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