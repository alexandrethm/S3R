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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# we choose the params of the following configuration
# grid_search_id 1129_1035, row 7 (91.63% mean_valid_acc when training on the SHREC dataset, with 14 gestures)
default_params = {
    'max_epochs': 2000,
    'batch_size': 64,
    'lr': 1e-4,
    'module__dropout': 0.4,
    'module__activation_fct': 'prelu',
    'module__preprocess': None,
    'module__conv_type': 'regular',

    # relation for 22 joints ? 132=6*22
    'module__channel_list': [
        (66, 3), (66, 3), (66, 3)
    ],

    # relation for 14 output gestures ? 70=5*14, 1400=100*14
    'module__fc_hidden_layers': [1400, 70],

    'module__temporal_attention': None,
}
grid_params = {
    'max_epochs': [2000],
    'batch_size': [8, 32, 64, 128, 256, 512],
    'lr': [1e-4],
    'module__dropout': [0.4],
    'module__activation_fct': ['prelu', 'relu'],
    'module__preprocess': [None],
    'module__conv_type': ['regular'],

    'module__channel_list': [
        [(66, 3)],
        [(66, 3), (66, 3)],
        [(66, 3), (66, 3), (66, 3)],
        [(66, 3), (66, 3), (66, 3), (66, 3)],
        [(66, 66), (66, 66), (66, 66)],
        [(66, 1), (66, 1), (66, 1)],
        [(66, 22), (66, 22), (66, 22)],
        [(22, 3), (22, 3), (22, 3)],
    ],
    # if preprocess: list of tuples [<(C_preprocess, None)>, (C_conv1, G_conv1), (C_conv2, G_conv2), (C_conv3, G_conv3), ...]
    # [(66, None), (66, 33), (66, 11)],
    # [(66, None), (66, 66), (66, 11)],
    # else: list of tuples [<(C_preprocess, None)>, (C_conv1, G_conv1), (C_conv2, G_conv2), (C_conv3, G_conv3), ...]
    # [(66,33), (66,11)],

    'module__fc_hidden_layers': [
        [2048, 256],
        [1024, 256],
        [512, 256],
        [256, 256],
        [2048, 128],
        [1024, 128],
        [512, 128],
        [256, 128],
        [128, 128],
        [128, 64],
    ],

    'module__temporal_attention': [None],  # can also be 'dot_attention', 'general_attention'
}


def get_configs():
    """
    Yields the param combinaisons specified in grid_params : change one parameter at a time (selected from the list in
    grid_params), while the other parameters are taken from the default_params dict

    :return: A generator that yields dictionaries of params
    """
    for param_key in grid_params:
        if len(grid_params[param_key]) > 1:
            # if there are non default configurations to try, try them
            for param in grid_params[param_key]:
                current_params = dict(default_params)
                current_params[param_key] = param
                yield current_params


# Other settings
temporal_seq_duration = 100  # how to resize time sequences
nb_output_gesture = 14  # 14 or 28
log_to_comet_ml = True

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

# unique identifier for the grid_search / random_search run
grid_id = 'grid_comparison_{:%m%d_%H%M}'.format(datetime.now())
# save every run final scores
scores = pd.DataFrame(columns=['acc', 'f1_score', 'cm'])

configs = get_configs()
for config in configs:
    # -------------
    # Create estimator
    # -------------
    net = NeuralNetClassifier(
        module=my_nets.Net,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        callbacks=[
            ('my_cb',
             code_S3R.utils.training_utils.S3RTrainingLoggerCallback(
                 param_keys_to_log=utils.get_param_keys(default_params),
                 search_run_id=grid_id,
                 log_to_comet_ml=log_to_comet_ml,
                 log_to_csv=False)),
            ('early_stopping', callbacks.EarlyStopping(patience=50))
        ],
        device=device
    )
    net.set_params(callbacks__print_log=None)  # deactivate default score printing each epoch
    net.set_params(**config)
    net.set_params(module__nb_classes=nb_output_gesture)

    # -------------
    # Fit estimator
    # -------------
    config_id = '{:%H%M%S}'.format(datetime.now())
    print('Starting to train config {0}'.format(config_id))
    print(str(config))
    net.fit(x_train, y_train)

    # -------------
    # Log metrics
    # -------------
    # save loss/acc plots
    history_df = pd.DataFrame(net.history).loc[:, ['epoch', 'dur', 'train_loss', 'valid_acc', 'valid_loss']]
    history_df.to_csv('results/comparisons/{0}_{1}_history.csv'.format(grid_id, config_id))

    # calculate and save metrics of the trained model
    y_pred = net.predict(x_test)
    scores = scores.append(pd.DataFrame({
        'config_id': config_id,
        'config_params': str(config),
        'acc': metrics.accuracy_score(y_test, y_pred),
        'f1_score': metrics.f1_score(y_test, y_pred, average='weighted'),
        'cm': [metrics.confusion_matrix(y_test, y_pred)]
    }))
    print(scores.iloc[-1,:]) # print last results

scores.to_csv('results/comparisons/{0}_scores.csv'.format(grid_id))
