import numpy as np
import torch
from sklearn.model_selection import *
from skorch import NeuralNetClassifier

from code_S3R import my_utils, my_nets

grid_search_params = [
    {
        'lr': [0.001], 'max_epochs': [200], 'batch_size': [32, 64],
        'module__activation_fct': ['relu', 'prelu', 'swish'],
    },
    {
        'lr': [0.0001], 'max_epochs': [500], 'batch_size': [32, 64],
        'module__activation_fct': ['relu', 'prelu', 'swish'],
    },
]

other_params = {
    'net_type': 'xyz',
    'temporal_duration': 100,
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
                                                                                         temporal_duration=other_params[
                                                                                             'temporal_duration'])

# Feeding it PyTorch tensors doesn't seem to work, but numpy arrays with the right format is okay
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
y_train_14 = y_train_14.astype(np.long)
y_test_14 = y_test_14.astype(np.long)
y_train_28 = y_train_28.astype(np.long)
y_test_28 = y_test_28.astype(np.long)

y_train = y_train_14
y_test = y_test_14

# -------------
# Perform grid search
# -------------
net = NeuralNetClassifier(
    module=my_nets.XYZNet,
    criterion=torch.nn.CrossEntropyLoss,
    optimizer=torch.optim.Adam,
    callbacks=[
        ('my_cb', my_utils.MyCallback(params_to_log=my_utils.get_param_keys(grid_search_params))),
    ],
)
net.set_params(callbacks__print_log=None)  # deactivate default score printing each epoch

gs = GridSearchCV(estimator=net, param_grid=grid_search_params, refit=False, scoring='accuracy', verbose=2)

gs.fit(x_train, y_train)

my_utils.save_and_print_results(grid_search=gs, grid_search_params=grid_search_params)
