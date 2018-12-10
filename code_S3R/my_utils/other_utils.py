import os
import os.path
import pathlib
from datetime import datetime

import numpy
import pandas
from comet_ml import Experiment
import torch
from scipy import ndimage, random
from sklearn.utils import shuffle
from skorch.callbacks import Callback
import warnings


# Hide warnings
def hide_scipy_zoom_warnings():
    warnings.filterwarnings('ignore', '.*output shape of zoom.*')


# Get hyper params
def get_channels_list(nb_configs, preprocessing):
    def _gcl(preprocessing):
        max_depth = 5

        hyper_params = []

        for depth in range(1, max_depth + 1):
            for G in [1, 3, 11, 22, 66]:

                channel_list = []

                if preprocessing:
                    # add preprocessing layer
                    upper_bound_preprocess = int(2 * 66 / G)
                    # Choose a random number of channels between 1 and 2*66, no matter the G
                    c = G * random.randint(1, upper_bound_preprocess + 1)
                    channel_list.append((c, None))

                # add convolution layers
                upper_bound_conv = int(2 * 66 / G)
                for layer in range(depth):
                    # Choose a random number of channels between 1 and 4*66, no matter the G
                    c = G * random.randint(1, upper_bound_conv + 1)
                    channel_list.append((c, G))

                hyper_params.append(channel_list)

        return hyper_params

    hyper_params_list = []
    for i in range(nb_configs):
        hyper_params_list += _gcl(preprocessing=preprocessing)

    return hyper_params_list


# Logging -------------

class MyCallback(Callback):
    """
    Log data at each run (hyper-parameters and metrics), and save it in a CSV file and/or in a comet_ml experiment.


    Args:
        param_keys_to_log (list): The names of the hyper-parameters to log
        search_run_id (str): Unique identifier or the grid search
        log_to_csv (bool):
        log_to_comet_ml (bool):

    """

    # place here attributes that are not fed as arguments to the __init__ method
    experiment = None
    params_to_log = None

    def __init__(self, param_keys_to_log, search_run_id, log_to_csv=True, log_to_comet_ml=True) -> None:
        self.param_keys_to_log = param_keys_to_log
        self.search_run_id = search_run_id
        self.log_to_csv = log_to_csv
        self.log_to_comet_ml = log_to_comet_ml

    def on_train_begin(self, net, **kwargs):
        """
        If log_to_comet_ml=True, create a comet.ml experiment and log hyper-parameters specified in params_to_log.

        If log_to_csv=True, create a new csv

        Args:
            net:
            **kwargs:

        """
        params = net.get_params()
        self.params_to_log = {}
        for key in self.param_keys_to_log:
            try:
                self.params_to_log[key] = params[key]
            except KeyError:
                # in case params[key] is not found (for some grid searches it can be the case)
                self.params_to_log[key] = None

        if self.log_to_comet_ml:
            self.experiment = Experiment(api_key=os.environ['COMET_ML_API_KEY'], project_name='S3R-V2')
            self.experiment.log_multiple_params(self.params_to_log)
            self.experiment.set_model_graph(net.__str__())

            # make it easier to regroup experiments by grid_search
            self.experiment.log_other('search_run_id', self.search_run_id)

    def on_epoch_end(self, net, **kwargs):
        """
        Log epoch metrics to comet.ml if required
        """
        data = net.history[-1]

        if self.log_to_comet_ml:
            self.experiment.log_multiple_metrics(
                dic=dict((key, data[key]) for key in [
                    'valid_acc',
                    'valid_loss',
                    'train_loss',
                ]),
                step=data['epoch']
            )

    def on_train_end(self, net, X=None, y=None, **kwargs):
        """
        Save the metrics in a csv file if required
        """
        if self.log_to_csv:
            valid_acc_data = net.history[:, 'valid_acc']
            valid_loss_data = net.history[:, 'valid_loss']
            train_loss_data = net.history[:, 'train_loss']

            run_results = pandas.DataFrame({
                'params': self.params_to_log,
                'data': {
                    'valid_acc': valid_acc_data,
                    'valid_loss': valid_loss_data,
                    'train_loss': train_loss_data,
                }
            })

            file_name = 'data_{:%H%M%S}.csv'.format(datetime.now())

            # create the directory if does not exist, without raising an error if it does already exist
            pathlib.Path('results/{}'.format(self.search_run_id)).mkdir(parents=True, exist_ok=True)

            # save the file
            run_results.to_csv(path_or_buf='results/{}/{}'.format(self.search_run_id, file_name))


def save_and_print_results(search_run_id, cv_results, grid_search_params):
    """
    Save results as a csv file.
    Print :
        - Best score with the corresponding params
        - Filename of the csv file
        - All the results just in case

    Args:
        grid_search_params:
        cv_results:
        search_run_id:
    """
    results = pandas.DataFrame(cv_results).sort_values('rank_test_score')

    # create the directory if does not exist, without raising an error if it does already exist
    pathlib.Path('results/{}'.format(search_run_id)).mkdir(parents=True, exist_ok=True)

    # select important columns to save
    columns = ['rank_test_score', 'mean_test_score', 'std_test_score']
    for key in get_param_keys(grid_search_params):
        columns.append('param_' + key)
    columns.append('mean_fit_time')

    # save important results
    results.to_csv(path_or_buf='results/{}/summary.csv'.format(search_run_id),
                   columns=columns)
    # save all search results, without excluding some columns
    results.to_csv(path_or_buf='results/{}/detailed.csv'.format(search_run_id))

    print('------')
    print('Results saved with search_run_id {}'.format(search_run_id))
    print('All results just in case :\n', cv_results)


# Loading and pre-processing -------------

def load_data(filepath='./data/data.numpy.npy'):
    """
    Returns hand gesture sequences (X) and their associated labels (Y).
    Each sequence has two different labels.
    The first label Y describes the gesture class out of 14 possible gestures (e.g. swiping your hand to the right).
    The second label Y describes the gesture class out of 28 possible gestures (e.g. swiping your hand to the right with
     your index pointed, or not pointed).
    """
    data = numpy.load(filepath, encoding='latin1')
    # data = [x_train, x_test, y_train_14, y_train_28, y_test_14, y_test_28]
    return data[0], data[1], data[2], data[3], data[4], data[5]


def resize_sequences_length(x_train, x_test, final_length=100):
    """
    Resize the time series by interpolating them to the same length
    """
    x_train = numpy.array([
        numpy.array([
            ndimage.zoom(x_i.T[j], final_length / len(x_i), mode='reflect') for j in
            range(numpy.size(x_i, 1))
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


def shuffle_dataset(x_train, x_test, y_train_14, y_train_28, y_test_14, y_test_28):
    """Shuffle the train/test data consistently."""
    # note: add random_state=0 for reproducibility
    x_train, y_train_14, y_train_28 = shuffle(x_train, y_train_14, y_train_28)
    x_test, y_test_14, y_test_28 = shuffle(x_test, y_test_14, y_test_28)
    return x_train, x_test, y_train_14, y_train_28, y_test_14, y_test_28


def preprocess_data(x_train, x_test, y_train_14, y_train_28, y_test_14, y_test_28, temporal_duration=100):
    """
    Preprocess the data as you want.
    """
    x_train, x_test, y_train_14, y_train_28, y_test_14, y_test_28 = shuffle_dataset(x_train, x_test, y_train_14,
                                                                                    y_train_28, y_test_14, y_test_28)
    x_train, x_test = resize_sequences_length(x_train, x_test, final_length=temporal_duration)

    y_train_14 = numpy.array(y_train_14)
    y_train_28 = numpy.array(y_train_28)
    y_test_14 = numpy.array(y_test_14)
    y_test_28 = numpy.array(y_test_28)

    # Remove 1 to all classes items (1-14 => 0-13 and 1-28 => 0-27)
    y_train_14 = y_train_14 - 1
    y_train_28 = y_train_28 - 1
    y_test_14 = y_test_14 - 1
    y_test_28 = y_test_28 - 1

    return x_train, x_test, y_train_14, y_train_28, y_test_14, y_test_28


def convert_to_pytorch_tensors(x_train, x_test, y_train_14, y_train_28, y_test_14, y_test_28):
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


def get_param_keys(params):
    """
    :param params: dict of parameters, or list containing dicts of parameters
    :return: all keys contained inside the dict or inside the dicts of the list
    """
    if params.__class__ == dict:
        params_keys = params.keys()
    elif params.__class__ == list:
        params_keys = []
        for i in range(len(params)):
            for key in params[i]:
                if key not in params_keys:
                    params_keys.append(key)
    else:
        params_keys = []

    return list(params_keys)


# Loading and pre-processing for Online DHG dataset -------------

def load_dataset_in_torch(use_online_dataset, use_14=True, return_both_14_and_28=False, train_dataset=True,
                          articulations='world', path_dataset='./data/'):
    """

    Args:
        use_online_dataset (bool):
        use_14 (bool):
        return_both_14_and_28 (bool):
        train_dataset (bool): If True, return the train dataset, else return the test dataset
        articulations (str): 'world', 'world_enhanced' (or 'simple' or 'image' ?) for online dataset.
          'world' (or 'image' ?) for standard dataset.
        path_dataset:


    Return format
    -----

    Type:
      torch.Tensor

    Shape:
      Default: (x, y)
      if return_both_14_and_28, return: (x, y14, y28)
      if segment_sequences, return: (x, y, start_end_frames)
      if return_both_14_and_28 and segment_sequences, return: (x, y14, y28, start_end_frames)
    """

    # standard dhg
    if use_online_dataset is False:

        all_labels_14 = torch.load(os.path.join(path_dataset, 'STANDARD_DHG__all_labels_14.pytorchdata'))
        all_labels_28 = torch.load(os.path.join(path_dataset, 'STANDARD_DHG__all_labels_28.pytorchdata'))

        if articulations == 'image':
            x_dataset = torch.load(os.path.join(path_dataset, 'STANDARD_DHG__all_skeletons_image.pytorchdata'))
        elif articulations == 'world':
            x_dataset = torch.load(os.path.join(path_dataset, 'STANDARD_DHG__all_skeletons_world.pytorchdata'))
        else:
            raise Exception('The dataset you asked for does not exist.')

        if use_14:
            y_dataset = all_labels_14
        else:
            y_dataset = all_labels_28

        if return_both_14_and_28:
            return x_dataset, all_labels_14, all_labels_28
        else:
            return x_dataset, y_dataset

    # online dhg
    if use_online_dataset is True:
        dataset_type = 'train' if train_dataset else 'test'

        all_labels_14 = torch.load(
            os.path.join(path_dataset, 'ONLINE_DHG__all_labels_14_{}.pytorchdata'.format(dataset_type)))
        all_labels_28 = torch.load(
            os.path.join(path_dataset, 'ONLINE_DHG__all_labels_28_{}.pytorchdata'.format(dataset_type)))

        if articulations == 'simple':
            x_dataset = torch.load(
                os.path.join(path_dataset, 'ONLINE_DHG__all_skeletons_{}.pytorchdata'.format(dataset_type)))
        elif articulations == 'image':
            x_dataset = torch.load(
                os.path.join(path_dataset, 'ONLINE_DHG__all_skeletons_image_{}.pytorchdata'.format(dataset_type)))
        elif articulations == 'world':
            x_dataset = torch.load(
                os.path.join(path_dataset, 'ONLINE_DHG__all_skeletons_world_{}.pytorchdata'.format(dataset_type)))
        elif articulations == 'world_enhanced':
            x_dataset = torch.load(
                os.path.join(path_dataset,
                             'ONLINE_DHG__all_skeletons_world_enhanced_{}.pytorchdata'.format(dataset_type)))
        else:
            raise Exception('The dataset you asked for does not exist.')

        if use_14:
            y_dataset = all_labels_14
        else:
            y_dataset = all_labels_28

        if return_both_14_and_28:
            return x_dataset, all_labels_14, all_labels_28
        else:
            return x_dataset, y_dataset


def load_unsequenced_test_dataset(path_dataset='./data/', enhanced=False):
    """

    Args:
        path_dataset:
        enhanced:

    Returns:
        (x_seq_list, y_14_seq_list, y_28_seq_list) - the list of unsliced sequences data
          and the list of unsliced sequences labels.

    """
    data_type = '_enhanced' if enhanced else ''

    x_seq_list = torch.load(
        os.path.join(path_dataset, 'ONLINE_DHG__all_skeletons_world{}_unsliced_test.pytorchdata'.format(data_type)))
    y_14_seq_list = torch.load(
        os.path.join(path_dataset, 'ONLINE_DHG__all_labels_14_unsliced_test.pytorchdata'))
    y_28_seq_list = torch.load(
        os.path.join(path_dataset, 'ONLINE_DHG__all_labels_28_unsliced_test.pytorchdata'))

    return x_seq_list, y_14_seq_list, y_28_seq_list


# (One-shot code) Used to load DHG / Online DHG from txt files and transform it to lists/Tensors -------------
def file_to_numpy(path):
    if os.path.exists(path):
        return numpy.array([s.replace('\n', '').split() for s in open(path, 'r').readlines()], dtype=numpy.float32)
    else:
        return None


def quicker_load_for_standard_dhg_dataset(root='/tmp/DHG2016dataset', root_out='./data/'):
    """
    Standard DHG dataset
    """
    """
    Standard DHG dataset
    """
    import os
    import numpy
    import torch

    n_gestures = 14
    n_fingers = 2
    n_subjects = 28
    n_essais = 5

    all_skeletons_image = []
    all_skeletons_world = []
    all_labels_14 = []
    all_labels_28 = []

    for gesture in range(1, n_gestures + 1):

        print('==> Gesture {}'.format(gesture))

        for finger in range(1, n_fingers + 1):
            for subject in range(1, n_subjects + 1):
                for essai in range(1, n_essais + 1):

                    skeletons_image = file_to_numpy(
                        root + '/gesture_{}/finger_{}/subject_{}/essai_{}/skeletons_image.txt'.format(gesture, finger,
                                                                                                      subject, essai))
                    skeletons_world = file_to_numpy(
                        root + '/gesture_{}/finger_{}/subject_{}/essai_{}/skeletons_world.txt'.format(gesture, finger,
                                                                                                      subject, essai))

                    if skeletons_world is not None:
                        skeletons_image = torch.from_numpy(skeletons_image)
                        skeletons_world = torch.from_numpy(skeletons_world)
                        all_skeletons_image.append(skeletons_image)
                        all_skeletons_world.append(skeletons_world)
                        all_labels_14.append(gesture)
                        all_labels_28.append(2 * gesture)

    print('Saving to disk...')
    torch.save(all_skeletons_image, root_out + '/STANDARD_DHG__all_skeletons_image.pytorchdata')
    torch.save(all_skeletons_world, root_out + '/STANDARD_DHG__all_skeletons_world.pytorchdata')
    torch.save(all_labels_14, root_out + '/STANDARD_DHG__all_labels_14.pytorchdata')
    torch.save(all_labels_28, root_out + '/STANDARD_DHG__all_labels_28.pytorchdata')
    print('Saved to disk.')


def get_seq_x(root, subject, sequence):
    """

    Args:
        root: Path of the root folder containing online dhg data
        subject:
        sequence:

    Returns:
        Torch tensors skeletons, skeletons_image, skeletons_world, skeletons_world_enhanced for the specified
        sequence and subject

    """
    skeletons = file_to_numpy(root + '/subject_{}/sequence_{}/skeletons.txt'.format(subject, sequence + 1))
    skeletons_image = file_to_numpy(root + '/subject_{}/sequence_{}/skeletons_image.txt'.format(subject, sequence + 1))
    skeletons_world = file_to_numpy(root + '/subject_{}/sequence_{}/skeletons_world.txt'.format(subject, sequence + 1))
    skeletons_world_enhanced = file_to_numpy(
        root + '/subject_{}/sequence_{}/skeletons_world_enhanced.txt'.format(subject, sequence + 1))

    skeletons = torch.from_numpy(skeletons)
    skeletons_image = torch.from_numpy(skeletons_image)
    skeletons_world = torch.from_numpy(skeletons_world)
    skeletons_world_enhanced = torch.from_numpy(skeletons_world_enhanced)

    return skeletons, skeletons_image, skeletons_world, skeletons_world_enhanced


def get_seq_y(gestures, fingers, time_bounds, seq_length):
    """

    Args:
        seq_length:
        gestures:
        fingers:
        time_bounds:

    Returns:
        Y labels for the whole sequence (with one Y label for each time step)

    """
    # Generate full Y data for the sequence
    assert len(gestures) == len(fingers) and 2 * len(gestures) == len(time_bounds)
    nb_gestures = len(gestures)

    y_seq_14 = torch.zeros(seq_length, 14 + 1, dtype=torch.int)  # +1 for the "no gesture" class
    y_seq_28 = torch.zeros(seq_length, 28 + 1, dtype=torch.int)

    # Labelize the 'real gestures' classes
    for g in range(nb_gestures):
        gesture = int(gestures[g])
        finger = int(fingers[g])
        t_start = int(time_bounds[2 * g])
        t_end = int(time_bounds[2 * g + 1])

        # Gestures classes go from 1 to 14, so we keep index 0 for the 'no gesture' class
        y_seq_14[t_start:t_end, gesture] = 1
        # Class is defined as follows : 2*k-1 for (finger=1, gesture=k) and class 2*k for (finger=2, gesture=k)
        # We go from index 1 (finger=1, gesture=1) to index 28 (finger=2, gesture=14)
        # and index 0 is kept for the 'no gesture class'
        y_seq_28[t_start:t_end, 2 * gesture + finger - 2] = 1

    # Labelize the 'no gesture' class
    for t in range(seq_length):
        # if there is no gesture at timestep t, add a 'no gesture' label
        if y_seq_14[t].sum().item() == 0:
            y_seq_14[t, 0] = 1
            y_seq_28[t, 0] = 1

    return y_seq_14, y_seq_28


def slice_sequence(full_ske, full_ske_im, full_x, full_x_enhanced, full_y_14, full_y_28, time_window, time_gap):
    """

    Slice a sequence of gestures performed by a subject into multiple windows of x/y data

    Args:
        full_ske:
        full_ske_im:
        full_x:
        full_x_enhanced:
        full_y_14:
        full_y_28:
        time_window:
        time_gap:

    Returns:
        The list of x and y data for the sequence

    """
    ske_list = []
    ske_im_list = []
    x_list = []
    x_enhanced_list = []
    y_list_14 = []
    y_list_28 = []

    assert full_x.shape[0] == full_x_enhanced.shape[0] \
           and full_x.shape[0] == full_ske.shape[0] \
           and full_x.shape[0] == full_ske_im.shape[0] \
           and full_x.shape[0] == full_y_14.shape[0] \
           and full_x.shape[0] == full_y_28.shape[0]
    seq_length, _ = full_x.shape

    # Slice the full sequences in multiple tensors of temporal duration 'time_window',
    # with 'time_gap' between the sequences
    t_a = 0
    t_b = t_a + time_window
    while t_b <= seq_length:
        # Take a slice of x from t_a (included) to t_b (excluded)
        ske = full_ske[t_a:t_b, :]
        ske_im = full_ske_im[t_a:t_b, :]
        x = full_x[t_a:t_b, :]
        x_enhanced = full_x_enhanced[t_a:t_b, :]

        # Define y for the slice as full_y[t_b - 1] : the networks's goal is to identify, in real time,
        # the gestures at time t given the preceding time steps
        y_14 = full_y_14[t_b - 1, :]
        y_28 = full_y_28[t_b - 1, :]

        # Add the new x/y to the list
        ske_list.append(ske)
        ske_im_list.append(ske_im)
        x_list.append(x)
        x_enhanced_list.append(x_enhanced)
        y_list_14.append(y_14)
        y_list_28.append(y_28)

        # Shift the time window
        t_a += time_gap
        t_b += time_gap

    return ske_list, ske_im_list, x_list, x_enhanced_list, y_list_14, y_list_28


def save_online_dhg_dataset(root='/Users/alexandre/Desktop/ODHG2016',
                            root_out='/Users/alexandre/Desktop/ODHG_torch_data', time_window=100, time_gap=10):
    """

    Load .txt files and save to disk x and y data, as torch tensors.

    Args:
        root:
        root_out:
        time_window:
        time_gap:

    """
    n_subjects = 28
    test_subjects_id = [0, 2, 3, 7, 12, 20, 21, 22]

    all_ske_train = []
    all_ske_im_train = []
    all_x_train = []
    all_x_enhanced_train = []
    all_y_14_train = []
    all_y_28_train = []

    all_ske_test = []
    all_ske_im_test = []
    all_x_test = []
    all_x_enhanced_test = []
    all_y_14_test = []
    all_y_28_test = []

    # For visualizing real time recognition
    all_x_unsliced_test = []
    all_x_enhanced_unsliced_test = []
    all_y_14_unsliced_test = []
    all_y_28_unsliced_test = []

    for subject in range(1, n_subjects + 1):

        print('==> Subject {}'.format(subject))

        # --------

        print('Load and create X/Y for each sequence...')
        subject_info_seq = [s.split() for s in
                            open(root + '/subject_{}_infos_sequences.txt'.format(subject), 'r').readlines()]

        # Make sure the file is made of 3 lines blocks
        assert len(subject_info_seq) % 3 == 0
        nb_seq = int(len(subject_info_seq) / 3)

        # Lists of X data for each sequence
        ske_seq_list = []
        ske_im_seq_list = []
        x_seq_list = []
        x_enhanced_seq_list = []

        # Lists of Y labels for each sequence
        y_seq_list_14 = []
        y_seq_list_28 = []

        for i in range(nb_seq):
            # For each sequences producted by the subject
            # Get X data
            skeletons, skeletons_image, skeletons_world, skeletons_world_enhanced = get_seq_x(root, subject=subject,
                                                                                              sequence=i)
            ske_seq_list.append(skeletons)
            ske_im_seq_list.append(skeletons_image)
            x_seq_list.append(skeletons_world)
            x_enhanced_seq_list.append(skeletons_world_enhanced)

            # Get Y data
            gestures = subject_info_seq[3 * i]
            fingers = subject_info_seq[3 * i + 1]
            time_bounds = subject_info_seq[3 * i + 2]
            y_seq_14, y_seq_28 = get_seq_y(gestures, fingers, time_bounds, seq_length=skeletons_world.shape[0])

            y_seq_list_14.append(y_seq_14)
            y_seq_list_28.append(y_seq_28)

        # --------

        print('Slice X and Y labels...')
        # Lists of X and Y data, after window slicing
        ske_list = []
        ske_im_list = []
        x_list = []
        x_enhanced_list = []
        y_list_14 = []
        y_list_28 = []

        for i in range(nb_seq):
            full_ske = ske_seq_list[i]
            full_ske_im = ske_im_seq_list[i]
            full_x = x_seq_list[i]
            full_x_enhanced = x_enhanced_seq_list[i]
            full_y_14 = y_seq_list_14[i]
            full_y_28 = y_seq_list_28[i]

            ske_list_sliced, ske_im_list_sliced, \
            x_list_sliced, x_enhanced_list_sliced, \
            y_list_14_sliced, y_list_28_sliced = slice_sequence(full_ske, full_ske_im,
                                                                full_x, full_x_enhanced,
                                                                full_y_14, full_y_28,
                                                                time_window, time_gap)

            ske_list += ske_list_sliced
            ske_im_list += ske_im_list_sliced
            x_list += x_list_sliced
            x_enhanced_list += x_enhanced_list_sliced
            y_list_14 += y_list_14_sliced
            y_list_28 += y_list_28_sliced

        print('Nb of x/y data after slicing (subject {}) : {}'.format(subject, len(x_list)))
        if subject in test_subjects_id:
            all_ske_test += ske_list
            all_ske_im_test += ske_im_list
            all_x_test += x_list
            all_x_enhanced_test += x_enhanced_list
            all_y_14_test += y_list_14
            all_y_28_test += y_list_28

            all_x_unsliced_test += x_seq_list
            all_x_enhanced_unsliced_test += x_enhanced_seq_list
            all_y_14_unsliced_test += y_seq_list_14
            all_y_28_unsliced_test += y_seq_list_28
        else:
            all_ske_train += ske_list
            all_ske_im_train += ske_im_list
            all_x_train += x_list
            all_x_enhanced_train += x_enhanced_list
            all_y_14_train += y_list_14
            all_y_28_train += y_list_28

    assert len(all_x_train) == len(all_ske_train) \
           and len(all_x_train) == len(all_ske_im_train) \
           and len(all_x_train) == len(all_x_enhanced_train) \
           and len(all_x_train) == len(all_y_14_train) \
           and len(all_x_train) == len(all_y_28_train)
    assert len(all_x_test) == len(all_ske_test) \
           and len(all_x_test) == len(all_ske_im_test) \
           and len(all_x_test) == len(all_x_enhanced_test) \
           and len(all_x_test) == len(all_y_14_test) \
           and len(all_x_test) == len(all_y_28_test)
    print('-' * 20)
    print('Total number of test data : {}'.format(len(all_x_test)))
    print('Total number of train data : {}'.format(len(all_x_train)))

    # Convert to torch tensors
    # Train data
    all_ske_train = torch.stack(all_ske_train)
    all_ske_im_train = torch.stack(all_ske_im_train)
    all_x_train = torch.stack(all_x_train)
    all_x_enhanced_train = torch.stack(all_x_enhanced_train)
    all_y_14_train = torch.stack(all_y_14_train)
    all_y_28_train = torch.stack(all_y_28_train)
    # Test data
    all_ske_test = torch.stack(all_ske_test)
    all_ske_im_test = torch.stack(all_ske_im_test)
    all_x_test = torch.stack(all_x_test)
    all_x_enhanced_test = torch.stack(all_x_enhanced_test)
    all_y_14_test = torch.stack(all_y_14_test)
    all_y_28_test = torch.stack(all_y_28_test)

    # Save to disk
    print('Saving to disk...')
    # create the directory if does not exist, without raising an error if it does already exist
    pathlib.Path(root_out).mkdir(parents=True, exist_ok=True)

    # Train data
    torch.save(all_ske_train, root_out + '/ONLINE_DHG__all_skeletons_train.pytorchdata')
    torch.save(all_ske_im_train, root_out + '/ONLINE_DHG__all_skeletons_image_train.pytorchdata')
    torch.save(all_x_train, root_out + '/ONLINE_DHG__all_skeletons_world_train.pytorchdata')
    torch.save(all_x_enhanced_train, root_out + '/ONLINE_DHG__all_skeletons_world_enhanced_train.pytorchdata')
    torch.save(all_y_14_train, root_out + '/ONLINE_DHG__all_labels_14_train.pytorchdata')
    torch.save(all_y_28_train, root_out + '/ONLINE_DHG__all_labels_28_train.pytorchdata')

    # Test data
    torch.save(all_ske_test, root_out + '/ONLINE_DHG__all_skeletons_test.pytorchdata')
    torch.save(all_ske_im_test, root_out + '/ONLINE_DHG__all_skeletons_image_test.pytorchdata')
    torch.save(all_x_test, root_out + '/ONLINE_DHG__all_skeletons_world_test.pytorchdata')
    torch.save(all_x_enhanced_test, root_out + '/ONLINE_DHG__all_skeletons_world_enhanced_test.pytorchdata')
    torch.save(all_y_14_test, root_out + '/ONLINE_DHG__all_labels_14_test.pytorchdata')
    torch.save(all_y_28_test, root_out + '/ONLINE_DHG__all_labels_28_test.pytorchdata')

    # Test data unsliced
    torch.save(all_x_unsliced_test, root_out + '/ONLINE_DHG__all_skeletons_world_unsliced_test.pytorchdata')
    torch.save(all_x_enhanced_unsliced_test,
               root_out + '/ONLINE_DHG__all_skeletons_world_enhanced_unsliced_test.pytorchdata')
    torch.save(all_y_14_unsliced_test, root_out + '/ONLINE_DHG__all_labels_14_unsliced_test.pytorchdata')
    torch.save(all_y_28_unsliced_test, root_out + '/ONLINE_DHG__all_labels_28_unsliced_test.pytorchdata')
    print('Saved to disk.')
