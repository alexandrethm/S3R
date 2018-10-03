import time

from comet_ml import Experiment

from code_S3R import my_utils

hyper_params = {
    'net_type': 'xyz',
    'activation_fct': 'relu',
    'temporal_duration': 100,

    'learning_rate': 1e-3,
    'num_epochs': 200,
    'batch_size': 32,
}

# -------------
# Data
# -------------

# Load the dataset
x_train, x_test, y_train_14, y_train_28, y_test_14, y_test_28 = my_utils.load_data()
# Shuffle sequences and resize sequences
x_train, x_test, y_train_14, y_train_28, y_test_14, y_test_28 = my_utils.preprocess_data(x_train, x_test,
                                                                                         y_train_14,
                                                                                         y_train_28,
                                                                                         y_test_14, y_test_28,
                                                                                         temporal_duration=hyper_params[
                                                                                             'temporal_duration'])
# Convert to pytorch variables
x_train, x_test, y_train_14, y_train_28, y_test_14, y_test_28 = my_utils.convert_to_pytorch_tensors(x_train, x_test,
                                                                                                    y_train_14,
                                                                                                    y_train_28,
                                                                                                    y_test_14,
                                                                                                    y_test_28)
y_train = y_train_14
y_test = y_test_14

# -------------
# Experiment
# -------------
time_tag = time.strftime('%d-%m_%H:%M:%S')
experiment_name = '{}/{}'.format('SHREC14', time_tag)

experiment = Experiment(api_key='Tz0dKZfqyBRMdGZe68FxU3wvZ', project_name='S3R')
experiment.log_multiple_params(hyper_params)
experiment.set_name(experiment_name)

my_utils.perform_training(x_train, y_train, x_test, y_test, hyper_params, experiment)
