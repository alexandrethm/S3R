from datetime import datetime

import numpy as np
import comet_ml
import torch
from scipy import stats
from sklearn.model_selection import *
from skorch import NeuralNetClassifier, callbacks

from code_S3R import my_nets
import code_S3R.my_utils.other_utils as utils

hyper_params = [
    {
        'max_epochs': [5], 'batch_size': [32],
        'lr': [0.0001],
        'module__preprocess': [None],
        'module__conv_type': ['temporal'],
        'module__channel_list': [
            [6],
        ],
        'module__activation_fct': ['prelu'],
        'module__dropout': [0.4],
    },
]

# -------------
# Data
# -------------

# Load the dataset
x_train, x_test, y_train_14, y_train_28, y_test_14, y_test_28 = utils.load_data()
# Shuffle sequences and resize sequences
x_train, x_test, y_train_14, y_train_28, y_test_14, y_test_28 = utils.preprocess_data(x_train, x_test,
                                                                                      y_train_14,
                                                                                      y_train_28,
                                                                                      y_test_14, y_test_28,
                                                                                      temporal_duration=100)

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# unique identifier for the grid_search / random_search run
search_run_id = 'grid_search_{:%m%d_%H%M}'.format(datetime.now())

net = NeuralNetClassifier(
    module=my_nets.Net,
    criterion=torch.nn.CrossEntropyLoss,
    optimizer=torch.optim.Adam,
    callbacks=[
        ('my_cb', utils.MyCallback(param_keys_to_log=utils.get_param_keys(hyper_params), search_run_id=search_run_id,
                                   log_to_comet_ml=False, log_to_csv=True)),
        ('early_stopping', callbacks.EarlyStopping(patience=50))
    ],
    device=device
)
net.set_params(callbacks__print_log=None)  # deactivate default score printing each epoch

gs = GridSearchCV(estimator=net, param_grid=hyper_params, refit=False, scoring='accuracy',
                  verbose=2, cv=3, error_score=0)

# gs = RandomizedSearchCV(estimator=net, param_distributions=hyper_params, refit=False, scoring='accuracy',
#                        verbose=2, cv=3, error_score=0)

gs.fit(x_train, y_train)

# -------------
# Save and log results
# -------------

utils.save_and_print_results(search_run_id=search_run_id, cv_results=gs.cv_results_, grid_search_params=hyper_params)
