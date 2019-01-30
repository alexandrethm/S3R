from datetime import datetime

import numpy as np
import comet_ml
import os
import torch
from scipy import stats
from sklearn.model_selection import *
from skorch import NeuralNetClassifier, callbacks, dataset
from skorch.dataset import CVSplit
from skorch.callbacks import EpochScoring

import code_S3R.utils.data_utils
from code_S3R import my_nets
import code_S3R.utils.other_utils as utils

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
import torch

class OnlineDHGDataset(torch.utils.data.Dataset):
    def __init__(self, train_set=True):
        x_train, y_train = code_S3R.utils.data_utils.load_dataset_in_torch(use_online_dataset=True, use_14=True,
                                                                           train_dataset=True)
        x_test, y_test = code_S3R.utils.data_utils.load_dataset_in_torch(use_online_dataset=True, use_14=True,
                                                                         train_dataset=False)
        y_train = y_train.type_as(torch.FloatTensor())
        y_test = y_test.type_as(torch.FloatTensor())
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.train_set = train_set

    def __len__(self):
        if self.train_set:
            return len(self.x_train)
        else:
            return len(self.x_test)

    def __getitem__(self, index):
        if self.train_set:
            return self.x_train[index], self.y_train[index]
        else:
            return self.x_test[index], self.y_test[index]


training_set = OnlineDHGDataset(train_set=True)
test_set = OnlineDHGDataset(train_set=False)

print(len(training_set))
training_generator = torch.utils.data.DataLoader(training_set, batch_size=64)
testing_generator = torch.utils.data.DataLoader(test_set, batch_size=64)

# Shuffle sequences and resize sequences
# x_train, x_test, y_train_14, y_train_28, y_test_14, y_test_28 = utils.preprocess_data(x_train, x_test,
#                                                                                       y_train_14,
#                                                                                       y_train_28,
#                                                                                       y_test_14, y_test_28,
#                                                                                       temporal_duration=100)

# Feeding it PyTorch tensors doesn't seem to work, but numpy arrays with the right format is okay
# x_train = x_train.astype(np.float32)
# x_test = x_test.astype(np.float32)
# y_train = y_train.astype(np.int64)
# y_test = y_test.astype(np.int64)

# -------------
# Perform grid search
# -------------


# # unique identifier for the grid_search / random_search run
# single_run_id = 'single_run_{:%m%d_%H%M}'.format(datetime.now())
#
# net = NeuralNetClassifier(
#     module=my_nets.Net,
#     max_epochs=2000, batch_size=64,
#     lr=0.0001,
#     module__preprocess=None,  # or None
#     module__conv_type='regular',
#     module__channel_list=[(22, 22)],
#     module__fc_hidden_layers=[1400, 42],
#     module__activation_fct='prelu',
#     module__dropout=0.4,
#     module__temporal_attention=None,
#     module__nb_classes=15,
#
#     criterion=torch.nn.MultiLabelSoftMarginLoss,
#     optimizer=torch.optim.Adam,
#     callbacks=[
#         ('my_cb', utils.MyCallback(param_keys_to_log=utils.get_param_keys(hyper_params),
#                                    search_run_id=single_run_id,
#                                    log_to_comet_ml=False, log_to_csv=True)),
#         ('early_stopping', callbacks.EarlyStopping(patience=50)),
#         # ('f1', EpochScoring('f1', name='f1', lower_is_better=False,))
#     ],
#     device=device,
#     train_split=CVSplit(cv=5, stratified=False),
# )
#
# net.set_params(callbacks__valid_acc=None)
#
# net.fit(x_train, y_train)
#
# params_file = open('results/{}/model_params.pkl'.format(single_run_id), 'wb')
# optimizer_file = open('results/{}/model_optimizer.pkl'.format(single_run_id), 'wb')
# net.save_params(f_params=params_file,
#                 f_optimizer=optimizer_file)
#
# # To check accuracy with the the test dataset
# y_proba = net.predict_proba(x_test)
# y_predict = net.predict(x_test)
# test_accuracy = np.mean(y_predict == y_test)
# print('Test accuracy : {}'.format(test_accuracy))


