from datetime import datetime

import numpy as np
import comet_ml
import torch
from scipy import stats
from sklearn.model_selection import *
from skorch import NeuralNetClassifier, callbacks, dataset

from code_S3R import my_nets
import code_S3R.my_utils.other_utils as utils

hyper_params = {
    'max_epochs': 2000, 'batch_size': 64,
    'lr': 0.0001,
    'preprocess': 'graph_conv',  # or None
    'conv_type': 'temporal',
    'channel_list': [(66, None), (66, 66)],
    # if preprocess: list of tuples [<(C_preprocess, None)>, (C_conv1, G_conv1), (C_conv2, G_conv2), (C_conv3, G_conv3), ...]
    # [(66, None), (66, 33), (66, 11)],
    # [(66, None), (66, 66), (66, 11)],
    # else: list of tuples [<(C_preprocess, None)>, (C_conv1, G_conv1), (C_conv2, G_conv2), (C_conv3, G_conv3), ...]
    # [(66,33), (66,11)],
    'fc_hidden_layers': [1936, 128],
    'activation_fct': 'prelu',
    'dropout': 0.4,
}

# -------------
# Data
# -------------
# keep quiet, scipy
utils.hide_scipy_zoom_warnings()

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
y_train_14 = y_train_14.astype(np.int64)
y_test_14 = y_test_14.astype(np.int64)
y_train_28 = y_train_28.astype(np.int64)
y_test_28 = y_test_28.astype(np.int64)

y_train = y_train_14
y_test = y_test_14

# -------------
# Perform grid search
# -------------

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# unique identifier for the grid_search / random_search run
single_run_id = 'single_run_{:%m%d_%H%M}'.format(datetime.now())

net = NeuralNetClassifier(
    module=my_nets.Net,
    max_epochs=2000, batch_size=64,
    lr=0.0001,
    module__preprocess=None,  # or None
    module__conv_type='regular',
    module__channel_list=[(132, 11), (22, 11)],
    module__fc_hidden_layers=[1400, 42],
    module__activation_fct='prelu',
    module__dropout=0.4,
    module__temporal_attention=None,

    criterion=torch.nn.CrossEntropyLoss,
    optimizer=torch.optim.Adam,
    callbacks=[
        ('my_cb', utils.MyCallback(param_keys_to_log=utils.get_param_keys(hyper_params),
                                   search_run_id=single_run_id,
                                   log_to_comet_ml=False, log_to_csv=False)),
        ('early_stopping', callbacks.EarlyStopping(patience=50))
    ],
    device=device,
)

net.fit(x_train, y_train_14)


# To check accuracy with the the test dataset
# y_proba = net.predict_proba(x_test)
# y_predict = net.predict(x_test)
# test_accuracy = np.mean(y_predict == y_test_14)
# print('Test accuracy : {}'.format(test_accuracy))
