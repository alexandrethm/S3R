# coding: utf-8

# In[1]:


from __future__ import unicode_literals, print_function, division
import torch
import numpy
import pickle
from scipy import ndimage as ndimage
from sklearn.utils import shuffle
import itertools
import time
import math

from tensorboard_logger import configure, log_value


# In[2]:


def load_data(filepath='/Users/alexandre/development/S3R/data/data.numpy.npy'):
    """
    Returns hand gesture sequences (X) and their associated labels (Y).
    Each sequence has two different labels.
    The first label Y describes the gesture class out of 14 possible gestures (e.g. swiping your hand to the right).
    The second label Y describes the gesture class out of 28 possible gestures (e.g. swiping your hand to the right with your index pointed, or not pointed).
    """
    data = numpy.load(filepath, encoding='latin1')
    # return data['x_train'], data['x_test'], data['y_train_14'], data['y_train_28'], data['y_test_14'], data['y_test_28']
    return data[0], data[1], data[2], data[3], data[4], data[5]


# In[3]:


def resize_sequences_length(x_train, x_test, final_length=100):
    """
    Resize the time series by interpolating them to the same length
    """
    # python2 important note: redefine the classic division operator / by importing it from the __future__ module
    x_train = numpy.array([
        numpy.array([
            ndimage.zoom(x_i.T[j], final_length / len(x_i), mode='reflect') for j in range(numpy.size(x_i, 1))
        ]).T
        for x_i
        in x_train
    ])
    x_test = numpy.array([
        numpy.array([
            ndimage.zoom(x_i.T[j], final_length / len(x_i), mode='reflect') for j in
            range(numpy.size(x_i, 1))
        ]).T
        for x_i
        in x_test
    ])
    return x_train, x_test


# In[4]:


def shuffle_dataset(x_train, x_test, y_train_14, y_train_28, y_test_14, y_test_28):
    """Shuffle the train/test data consistently."""
    # note: add random_state=0 for reproducibility
    x_train, y_train_14, y_train_28 = shuffle(x_train, y_train_14, y_train_28)
    x_test, y_test_14, y_test_28 = shuffle(x_test, y_test_14, y_test_28)
    return x_train, x_test, y_train_14, y_train_28, y_test_14, y_test_28


# In[5]:


def preprocess_data(x_train, x_test, y_train_14, y_train_28, y_test_14, y_test_28):
    """
    Preprocess the data as you want.
    """
    x_train, x_test, y_train_14, y_train_28, y_test_14, y_test_28 = shuffle_dataset(x_train, x_test, y_train_14,
                                                                                    y_train_28, y_test_14, y_test_28)
    x_train, x_test = resize_sequences_length(x_train, x_test, final_length=100)

    return x_train, x_test, y_train_14, y_train_28, y_test_14, y_test_28


# In[6]:


def convert_to_pytorch_tensors(x_train, x_test, y_train_14, y_train_28, y_test_14, y_test_28):
    y_train_14 = numpy.array(y_train_14)
    y_train_28 = numpy.array(y_train_28)
    y_test_14 = numpy.array(y_test_14)
    y_test_28 = numpy.array(y_test_28)

    # Remove 1 to all classes items (1-14 => 0-13 and 1-28 => 0-27)
    y_train_14 = y_train_14 - 1
    y_train_28 = y_train_28 - 1
    y_test_14 = y_test_14 - 1
    y_test_28 = y_test_28 - 1

    x_train = torch.from_numpy(x_train)
    x_test = torch.from_numpy(x_test)
    y_train_14 = torch.from_numpy(y_train_14)
    y_train_28 = torch.from_numpy(y_train_28)
    y_test_14 = torch.from_numpy(y_test_14)
    y_test_28 = torch.from_numpy(y_test_28)

    x_train = x_train.type(torch.FloatTensor)
    x_test = x_test.type(torch.FloatTensor)
    y_train_14 = y_train_14.type(torch.LongTensor)
    y_test_14 = y_test_14.type(torch.LongTensor)
    y_train_28 = y_train_28.type(torch.LongTensor)
    y_test_28 = y_test_28.type(torch.LongTensor)

    return x_train, x_test, y_train_14, y_train_28, y_test_14, y_test_28


# In[7]:


class Net(torch.nn.Module):
    """
    [Devineau et al., 2018] Deep Learning for Hand Gesture Recognition on Skeletal Data
    
    This model computes a succession of 3x [convolutions and pooling] independently on each of the 66 sequence channels.
    
    Each of these computations are actually done at two different resolutions, that are later merged by concatenation
    with the (pooled) original sequence channel.
    
    Finally, a multi-layer perceptron merges all of the processed channels and outputs a classification.
    
    In short:
    
        1. input --> split into 66 channels
        
        2.1. channel_i --> 3x [conv/pool/dropout] low_resolution_i
        2.2. channel_i --> 3x [conv/pool/dropout] high_resolution_i
        2.3. channel_i --> pooled_i
        
        2.4. low_resolution_i, high_resolution_i, pooled_i --> output_channel_i
        
        3. MLP(66x [output_channel_i]) ---> classification
    
    
    Note: "joint" is a synonym of "channel".
    
    """

    def __init__(self, num_joints=66, num_classes=14):

        """
        Instantiation of the parameters.
        
        Params:
            :param num_joints: 
            :param num_classes: 
        
        """

        super(Net, self).__init__()

        self.dropout_probability = 0.2

        # Layers ----------------------------------------------
        self.all_conv_high_res_first = torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=8, kernel_size=7, padding=3),
            torch.nn.ReLU(),
            torch.nn.AvgPool1d(2)
        )])
        self.all_conv_high_res_then = torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=8, out_channels=4, kernel_size=7, padding=3),
            torch.nn.ReLU(),
            torch.nn.AvgPool1d(2),

            torch.nn.Conv1d(in_channels=4, out_channels=4, kernel_size=7, padding=3),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=self.dropout_probability),
            torch.nn.AvgPool1d(2)
        ) for joint in range(num_joints + 1)])

        self.all_conv_low_first = torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AvgPool1d(2)
        )])
        self.all_conv_low_res_then = torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=8, out_channels=4, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AvgPool1d(2),

            torch.nn.Conv1d(in_channels=4, out_channels=4, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=self.dropout_probability),
            torch.nn.AvgPool1d(2)
        ) for joint in range(num_joints + 1)])

        self.all_residual = torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.AvgPool1d(2),
            torch.nn.AvgPool1d(2),
            torch.nn.AvgPool1d(2)
        ) for joint in range(num_joints + 1)])

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=9 * 66 * 12, out_features=1936),
            # <-- depends of the sequences lengths (cf. below)
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=1936, out_features=num_classes)
        )

        # Initialization --------------------------------------
        # Xavier init
        for module in itertools.chain(self.all_conv_high_res_first, self.all_conv_high_res_then,
                                      self.all_conv_low_first, self.all_conv_low_res_then, self.all_residual):
            for layer in module:
                if layer.__class__.__name__ == "Conv1d":
                    torch.nn.init.xavier_uniform(layer.weight, gain=torch.nn.init.calculate_gain('relu'))
                    torch.nn.init.constant(layer.bias, 0.1)

        for layer in self.fc:
            if layer.__class__.__name__ == "Linear":
                torch.nn.init.xavier_uniform(layer.weight, gain=torch.nn.init.calculate_gain('relu'))
                torch.nn.init.constant(layer.bias, 0.1)

    def forward(self, input):
        """
        This function performs the actual computations of the network for a forward pass.

        Params:
            :param input: A tensor of gestures. Its shape is (batch_size x temporal_duration x nb_channels)
        """

        # BS, TS, SS
        _, _, spatial_steps = input.size()

        # Work on each joint separately
        every_joint_out_all_time = []

        for joint in range(0, spatial_steps):
            input_joint = input[:, :, joint]

            # Add a dummy (spatial) dimension for the time convolutions
            # Conv1D format : (batch_size, num_feature_maps, length_of_seq)
            input_joint = input_joint.unsqueeze(1)

            # note: the first conv/pool weigths are shared between all channels
            high = self.all_conv_high_res_first[0](input_joint)
            high = self.all_conv_high_res_then[joint](high)

            low = self.all_conv_low_first[0](input_joint)
            low = self.all_conv_low_res_then[joint](low)

            orig = self.all_residual[joint](input_joint)

            # Time convolutions are concatenated along the feature maps axis
            output_joint = torch.cat([
                high,
                low,
                orig
            ], dim=1)
            every_joint_out_all_time.append(output_joint)

        # Concatenate along the feature maps axis
        every_joint_out_all_time_cat = torch.cat(every_joint_out_all_time, dim=1)

        # Flatten for the Linear layers
        every_joint_out_all_time_cat = every_joint_out_all_time_cat.view(-1,
                                                                         9 * 66 * 12)  # <-- depends of the sequences lengths
        # 9 * 66 * 12 = numpy.product(list(every_joint_out_all_time_cat.size()))/32, where 32 is the batch size

        # Fully-Connected Layers
        output = self.fc(every_joint_out_all_time_cat)

        return output


# In[8]:


# -------------
# Misc.
# -------------


def batch(tensor, batch_size=32):
    """Return a list of (mini) batches"""
    tensor_list = []
    length = tensor.shape[0]
    i = 0
    while True:
        if (i + 1) * batch_size >= length:
            tensor_list.append(tensor[i * batch_size: length])
            return tensor_list
        tensor_list.append(tensor[i * batch_size: (i + 1) * batch_size])
        i += 1


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '{:02d}m {:02d}s'.format(int(m), int(s))


def get_accuracy(model, x_tensor, y_tensor_real):
    """
    Params:
        :param model: a **pytorch** model
        :param x_tensor:  a **pytorch** tensor/Variable
        :param y_tensor_real:  a **pytorch** tensor/Variable
    """
    with torch.no_grad():
        model.eval()
        predicted = model(x_tensor)
        predicted = predicted.max(dim=1)[1]
        s = (predicted.long() == y_tensor_real.long()).sum().item()
        s = 1.0 * s / y_tensor_real.size()[0]
        model.train()
        return s


# In[9]:


# -------------
# Data
# -------------

# Load the dataset
x_train, x_test, y_train_14, y_train_28, y_test_14, y_test_28 = load_data()

# Shuffle sequences and resize sequences
x_train, x_test, y_train_14, y_train_28, y_test_14, y_test_28 = preprocess_data(x_train, x_test, y_train_14, y_train_28,
                                                                                y_test_14, y_test_28)

# Convert to pytorch variables
x_train, x_test, y_train_14, y_train_28, y_test_14, y_test_28 = convert_to_pytorch_tensors(x_train, x_test, y_train_14,
                                                                                           y_train_28, y_test_14,
                                                                                           y_test_28)

# In[10]:


# -------------
# Network instantiation
# -------------
net = Net(num_classes=14)

# In[11]:


# -----------------------------------------------------
# 3. Loss function & Optimizer
# -----------------------------------------------------
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=net.parameters(), lr=1e-3)


# In[13]:


# -------------
# Training
# -------------


def train(model, criterion, optimizer,
          x_train, y_train, x_test, y_test,
          num_epochs=5):
    # Prepare all mini-batches
    x_train_batches = batch(x_train)
    y_train_batches = batch(y_train)

    # Training starting time
    start = time.time()

    print('[INFO] Started to train the model.')

    # For TensorBoard visualization
    configure("../../runs/run-regular-initial", flush_secs=5)

    for ep in range(num_epochs):

        # Ensure we're still in training mode
        model.train()

        current_loss = 0.0

        for idx_batch, train_batches in enumerate(zip(x_train_batches, y_train_batches)):
            # get a mini-batch of sequences
            x_train_batch, y_train_batch = train_batches
            # x_train_batch, y_train_batch = torch.autograd.Variable(x_train_batch), torch.autograd.Variable(y_train_batch)

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

        accuracy_train = get_accuracy(model, x_train, y_train)
        accuracy_test = get_accuracy(model, x_test, y_test)
        print(
            'Epoch #{:03d} | Time elapsed : {} | Loss : {:.4e} | Accuracy_train : {:.4e} | Accuracy_test : {:.4e}'.format(
                ep + 1, time_since(start), current_loss, accuracy_train, accuracy_test))
        log_value('Loss', current_loss, ep)
        log_value('Accuracy_train', accuracy_train, ep)
        log_value('Accuracy_test', accuracy_test, ep)

    print('[INFO] Finished training the model. Total time : {}.'.format(time_since(start)))


# In[14]:


train(model=net, criterion=criterion, optimizer=optimizer,
      x_train=x_train, y_train=y_train_14, x_test=x_test, y_test=y_test_14,
      num_epochs=20)
