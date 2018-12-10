"""
Usage:
  run_online_step_1.py [--window_length=<window_length>] [--batch_size=<batch_size>] [--epochs=<epochs>] [--lr=<lr>]
  run_online_step_1.py -h | --help

Options:
  -h --help                        Show this screen.
  --window_length=<window_length>  Window length [default: 100].
  --batch_size=<batch_size>        Batch size [default: 64].
  --epochs=<epochs>                Training epochs [default: 200].
  --lr=<lr>                        Learning rate [default: 1e-4].

"""
from docopt import docopt
import numpy as np
import comet_ml
import os
import torch
from code_S3R import my_nets
import code_S3R.utils.other_utils as utils
import matplotlib.pyplot as plt
import seaborn
import numpy as np
from code_S3R.utils.data_utils import OnlineDHGDataset, load_unsequenced_test_dataset
from torch.nn.functional import softmax

# keep quiet, scipy
utils.hide_scipy_zoom_warnings()

# comet ml
os.environ['COMET_ML_API_KEY'] = 'Tz0dKZfqyBRMdGZe68FxU3wvZ'

# -------------
# Arguments
# -------------
arguments = docopt(__doc__, version='0.1.1rc')
batch_size = int(arguments['--batch_size'])
window_length = int(arguments['--window_length'])
epochs = int(arguments['--epochs'])
lr = float(arguments['--lr'])


# -------------
# Data
# -------------
# Load the dataset
training_generator = torch.utils.data.DataLoader(OnlineDHGDataset(set='train'), batch_size=batch_size)
validation_generator = torch.utils.data.DataLoader(OnlineDHGDataset(set='validation'), batch_size=batch_size)
testing_generator = torch.utils.data.DataLoader(OnlineDHGDataset(set='test'), batch_size=batch_size)

x_test_list, y_test_list = load_unsequenced_test_dataset()
x_test_list = [x.cuda() for x in x_test_list]
y_test_list = [y.cuda() for y in y_test_list]
y_test_list = [y[window_length:] for y in y_test_list]

# -------------
# Model
# -------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device is 'cuda':
    torch.backends.cudnn.benchmark = True
model = my_nets.Net(preprocess=None,
                    conv_type='regular',
                    channel_list=[(22, 22)],
                    temporal_attention=None,
                    fc_hidden_layers=[1400, 42],
                    sequence_length=window_length,
                    activation_fct='prelu',
                    dropout=0.4,
                    nb_classes=15)
model = model.cuda(device=device)
print('Created model')

# -------------
# Training
# -------------
optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
criterion = torch.nn.MultiLabelSoftMarginLoss()

print('Started training')
experiment = comet_ml.Experiment(api_key=os.environ['COMET_ML_API_KEY'], project_name='S3R-V_online')
experiment.log_other('lr', lr)
experiment.log_other('window_length', window_length)
for ep in range(epochs):

    train_loss_ep = 0
    valid_loss_ep = 0

    # --- Training ---
    model.train()
    for i, (batch_x, batch_y) in enumerate(training_generator):

        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        if window_length > batch_x.shape[1]:
            raise Exception('You want to classify sequences shorter than the size of your window ! Change the data!')
        if window_length != batch_x.shape[1]:
            batch_x = batch_x[:, -window_length:, :]

        optimizer.zero_grad()

        out = model(batch_x)
        loss = criterion(out, batch_y)

        train_loss_ep += loss.item()

        loss.backward()
        optimizer.step()

    # --- Validation ---
    model.eval()
    for i, (batch_x, batch_y) in enumerate(validation_generator):

        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        if window_length > batch_x.shape[1]:
            raise Exception('You want to classify sequences shorter than the size of your window ! Change the data!')
        if window_length != batch_x.shape[1]:
            batch_x = batch_x[:, -window_length:, :]

        with torch.no_grad():
            out = model(batch_x)
            loss = criterion(out, batch_y)
            valid_loss_ep += loss.item()

    train_loss_ep /= len(training_generator)
    valid_loss_ep /= len(validation_generator)

    experiment.log_metric(name='train_loss', value=train_loss_ep, step=ep)
    experiment.log_metric(name='valid_loss', value=valid_loss_ep, step=ep)

    # make it easier to regroup experiments by grid_search
    experiment.log_other('search_run_id', 'test dataset online dhg')

    print('epoch {} -- training loss = {} -- valid loss = {}'.format(ep, train_loss_ep, valid_loss_ep))

    # Checkpoints
    if ep % 100 == 0:
        torch.save(model, f='./results/model__checkpoint__window_{}_ep_{}.cudapytorchmodel'.format(window_length, ep))

print('Ended training')

torch.save(model, f='./results/model__window_{}.cudapytorchmodel'.format(window_length))
print('Saved model to disk')

print('Successfully finished. You can now run :')
print('  $  pipenv run run_online_step_2.py')
