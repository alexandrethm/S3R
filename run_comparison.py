from datetime import datetime

import pandas as pd
import numpy as np
import comet_ml
import torch
from sklearn import metrics
from skorch import NeuralNetClassifier
from skorch import callbacks

import code_S3R.utils.data_utils_numpy
import code_S3R.utils.training_utils
from code_S3R import my_nets
import code_S3R.utils.other_utils as utils

# we choose the params of the following configuration
# grid_search_id 1129_1035, row 7 (91.63% mean_valid_acc when training on the SHREC dataset, with 14 gestures)
default_params = {
    'max_epochs': 2000,
    'batch_size': 64,
    'lr': 10e-4,
    'module__dropout': 0.4,
    'module__activation_fct': 'prelu',
    'module__preprocess': None,
    'module__conv_type': 'regular',

    # relation for 22 joints ? 132=6*22
    'module__channel_list': [(132, 11), (22, 11)],

    # relation for 14 output gestures ? 70=5*14, 1400=100*14
    'module__fc_hidden_layers': [1400, 70],

    'module__temporal_attention': None,
}

# Other settings
temporal_seq_duration = 100  # how to resize time sequences
nb_output_gesture = 14  # 14 or 28
log_to_comet_ml = False

# -------------
# Load the dataset
# -------------
x_train, x_test, y_train_14, y_train_28, y_test_14, y_test_28 = code_S3R.utils.data_utils_numpy.load_data()
# Shuffle sequences and resize sequences
x_train, x_test, y_train_14, y_train_28, y_test_14, y_test_28 = code_S3R.utils.data_utils_numpy.preprocess_data(x_train,
                                                                                                                x_test,
                                                                                                                y_train_14,
                                                                                                                y_train_28,
                                                                                                                y_test_14,
                                                                                                                y_test_28,
                                                                                                                temporal_duration=temporal_seq_duration)

# Feeding it PyTorch tensors doesn't seem to work, but numpy arrays with the right format is okay
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
y_train_14 = y_train_14.astype(np.int64)
y_test_14 = y_test_14.astype(np.int64)
y_train_28 = y_train_28.astype(np.int64)
y_test_28 = y_test_28.astype(np.int64)

y_train, y_test = (y_train_14, y_test_14) if nb_output_gesture == 14 else (y_train_28, y_test_28)

# -------------
# Create estimator
# -------------

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# unique identifier for the grid_search / random_search run
grid_id = 'grid_comparison_{:%m%d_%H%M}'.format(datetime.now())

net = NeuralNetClassifier(
    module=my_nets.Net,
    criterion=torch.nn.CrossEntropyLoss,
    optimizer=torch.optim.Adam,
    callbacks=[
        ('my_cb',
         code_S3R.utils.training_utils.S3RTrainingLoggerCallback(param_keys_to_log=utils.get_param_keys(default_params),
                                                                 search_run_id=grid_id,
                                                                 log_to_comet_ml=log_to_comet_ml,
                                                                 log_to_csv=False)),
        ('early_stopping', callbacks.EarlyStopping(patience=50))
    ],
    device=device
)
# net.set_params(callbacks__print_log=None)  # deactivate default score printing each epoch
net.set_params(**default_params)
net.set_params(module__nb_classes=nb_output_gesture)

# save every run final scores
scores = pd.DataFrame(columns=['acc', 'f1_score', 'cm'])

# todo: begin for loop
# -------------
# Fit estimator
# -------------
net.fit(x_train, y_train)

# -------------
# Log metrics
# -------------
fit_id = 'default'

# save loss/acc plots
history_df = pd.DataFrame(net.history)['epoch', 'dur', 'train_loss', 'valid_acc', 'valid_loss']
history_df.to_csv('results/comparisons/{0}_{1}_history.csv'.format(grid_id, fit_id))

# calculate and save metrics of the trained model
y_pred = net.predict(x_test)
scores = scores.append(pd.DataFrame({
    'acc': metrics.accuracy_score(y_test, y_pred),
    'f1_score': metrics.f1_score(y_test, y_pred, average='weighted'),
    'cm': [metrics.confusion_matrix(y_test, y_pred)]
}))
# todo: end for loop

scores.to_csv('results/comparisons/{0}_scores.csv'.format(grid_id))
