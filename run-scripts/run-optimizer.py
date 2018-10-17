import numpy as np
import torch
import comet_ml
from sklearn.model_selection import *

from code_S3R import my_utils, my_nets

hyper_params = {
    'net_type': 'xyz',
    'temporal_duration': 100,
}

grid_search_params = {
    'lr': [1e-3, 1e-2],
    'max_epochs': [10],
    'batch_size': [16, 32],

    'module__activation_fct': ['relu', 'prelu', 'swish'],
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

net = my_utils.MyNeuralNetClassifier(
    module=my_nets.XYZNet,
    criterion=torch.nn.CrossEntropyLoss,
    optimizer=torch.optim.Adam,
    callbacks=[
        ('my_cb', my_utils.MyCallback()),
    ],
    keys_to_log=list(grid_search_params.keys()),
)

gs = RandomizedSearchCV(estimator=net, param_distributions=grid_search_params, refit=False, scoring='accuracy',
                        verbose=2)

gs.fit(x_train, y_train)
print(gs.best_score_, gs.best_params_)
print(gs.cv_results_)
