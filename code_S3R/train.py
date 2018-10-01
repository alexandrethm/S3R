from __future__ import unicode_literals, print_function, division

import time

import torch
from tensorboard_logger import configure, log_value

import code_S3R.nets as my_nets
import code_S3R.utility_functions as my_utils


# -------------
# Training
# -------------

def train(model, criterion, optimizer,
          x_train, y_train, x_test, y_test,
          num_epochs=5):
    # Prepare all mini-batches
    x_train_batches = my_utils.batch(x_train)
    y_train_batches = my_utils.batch(y_train)

    # Training starting time
    start = time.time()

    print('[INFO] Started to train the model.')

    # For TensorBoard visualization
    if net.__class__ == my_nets.RegularNet:
        net_type = 'regular'
    elif net.__class__ == my_nets.XYZNet:
        net_type = 'xyz'
    else:
        net_type = ''
    time_tag = time.strftime('%d-%m_%H:%M:%S')

    logdir = '../runs/{}/{}/{}'.format(net_type, net.activation_fct, time_tag)
    configure(logdir, flush_secs=5)

    for ep in range(num_epochs):

        # Ensure we're still in training mode
        model.train()

        current_loss = 0.0

        for idx_batch, train_batches in enumerate(zip(x_train_batches, y_train_batches)):
            # get a mini-batch of sequences
            x_train_batch, y_train_batch = train_batches

            # zero the gradient parameters
            optimizer.zero_grad()

            # forward
            outputs = model(x_train_batch)

            # backward + optimize
            # backward
            loss = criterion(outputs, y_train_batch)
            loss.backward()
            # optimize
            optimizer.step()
            # for an easy access
            current_loss += loss.item()

        accuracy_train = my_utils.get_accuracy(model, x_train, y_train)
        accuracy_test = my_utils.get_accuracy(model, x_test, y_test)
        print(
            'Epoch #{:03d} | Time elapsed : {} | Loss : {:.4e} | Accuracy_train : {:.4e} | Accuracy_test : {:.4e}'.format(
                ep + 1, my_utils.time_since(start), current_loss, accuracy_train, accuracy_test))
        log_value('Loss', current_loss, ep)
        log_value('Accuracy_train', accuracy_train, ep)
        log_value('Accuracy_test', accuracy_test, ep)

    print('[INFO] Finished training the model. Total time : {}.'.format(my_utils.time_since(start)))


# -------------
# Data
# -------------

# Load the dataset

x_train, x_test, y_train_14, y_train_28, y_test_14, y_test_28 = my_utils.load_data()

# Shuffle sequences and resize sequences
x_train, x_test, y_train_14, y_train_28, y_test_14, y_test_28 = my_utils.preprocess_data(x_train, x_test,
                                                                                         y_train_14,
                                                                                         y_train_28,
                                                                                         y_test_14, y_test_28)

# Convert to pytorch variables
x_train, x_test, y_train_14, y_train_28, y_test_14, y_test_28 = my_utils.convert_to_pytorch_tensors(x_train, x_test,
                                                                                                    y_train_14,
                                                                                                    y_train_28,
                                                                                                    y_test_14,
                                                                                                    y_test_28)

# -------------
# Network instantiation
# -------------

regular_net = my_nets.RegularNet()
xyz_net = my_nets.XYZNet(activation_fct='prelu')

net = xyz_net

# -----------------------------------------------------
# Loss function & Optimizer
# -----------------------------------------------------

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=net.parameters())

train(model=net, criterion=criterion, optimizer=optimizer,
      x_train=x_train, y_train=y_train_14, x_test=x_test, y_test=y_test_14,
      num_epochs=200)
