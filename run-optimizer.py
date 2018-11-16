import numpy as np
import comet_ml
import torch
from scipy import stats
from sklearn.model_selection import *
from skorch import NeuralNetClassifier, callbacks

from code_S3R import my_nets
import code_S3R.my_utils.other_utils as utils

# grid_search_params = [
#     {
#         'max_epochs': [1000], 'batch_size': [32],
#         'lr': [0.0001],
#         'module__dropout': [0.4],
#         'module__activation_fct': ['prelu'],
#         'module__net_type': ['TCN'],
#         'module__tcn_channels': [
#             [33, 15, 5, 3],
#             [33, 15, 5, 3, 1]
#         ]
#     }
# ]
hyper_params = {
    'max_epochs': [1000], 'batch_size': [32],
    'lr': [0.0001],
    'module__dropout': stats.uniform(0, 1),
    'module__activation_fct': ['prelu'],
    'module__net_type': ['TCN'],
    'module__tcn_channels': [
        [22, 11],
        [22, 6],
        [33, 22],
        [33, 11, 11],
        [22, 22, 11],
    ],
    # 'module__tcn_k': stats.randint(2, 5)
}
my_net = my_nets.TCN if hyper_params['module__net_type'] == ['TCN'] else my_nets.Net

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

net = NeuralNetClassifier(
    module=my_net,
    criterion=torch.nn.CrossEntropyLoss,
    optimizer=torch.optim.Adam,
    callbacks=[
        ('my_cb', utils.MyCallback(params_to_log=utils.get_param_keys(hyper_params))),
        ('early_stopping', callbacks.EarlyStopping(patience=50))
    ],
)
net.set_params(callbacks__print_log=None)  # deactivate default score printing each epoch

gs = RandomizedSearchCV(estimator=net, param_distributions=hyper_params, refit=False, scoring='accuracy',
                        verbose=2, cv=3, error_score=0)

gs.fit(x_train, y_train)

# -------------
# Save and log results
# -------------

utils.save_and_print_results(cv_results=gs.cv_results_, grid_search_params=hyper_params)
