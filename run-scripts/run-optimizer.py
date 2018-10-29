import numpy as np
import torch
import comet_ml
from sklearn.model_selection import *
from skorch import NeuralNetClassifier

from code_S3R import my_utils, my_nets

grid_search_params = [
    {
        'max_epochs': [600], 'batch_size': [8, 32],
        'lr': [0.0001],
        'module__dropout': [0.1],
        'module__activation_fct': ['prelu'],
    },
    {
        'max_epochs': [800], 'batch_size': [8, 32],
        'lr': [0.0001],
        'module__dropout': [0.4],
        'module__activation_fct': ['prelu'],
    },
    {
        'max_epochs': [1000], 'batch_size': [128],
        'lr': [0.0001],
        'module__dropout': [0.1, 0.2, 0.4],
        'module__activation_fct': ['prelu'],
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

gs = GridSearchCV(estimator=net, param_grid=grid_search_params, refit=False, scoring='accuracy', verbose=2,
                  error_score=0)

gs.fit(x_train, y_train)

# -------------
# Save and log results
# -------------

my_utils.save_and_print_results(cv_results=gs.cv_results_, grid_search_params=grid_search_params)
