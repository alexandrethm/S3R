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
    The second label Y describes the gesture class out of 28 possible gestures (e.g. swiping your hand to the right with your index pointed, or not pointed).
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

def load_dataset_in_torch(use_online_dataset=False, use_14=True, return_both_14_and_28=False, articulations='world',
                          segment_sequences=False, path_dataset='./data/'):
    """
    Return format
    -------------

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

        all_labels_14 = torch.load(os.path.join(path_dataset, 'ONLINE_DHG__all_labels_14.pytorchdata'))
        all_labels_28 = torch.load(os.path.join(path_dataset, 'ONLINE_DHG__all_labels_28.pytorchdata'))

        all_start_end_frames = torch.load(os.path.join(path_dataset, 'ONLINE_DHG__all_start_end_frames.pytorchdata'))

        if articulations == 'simple' and not segment_sequences:
            x_dataset = torch.load(os.path.join(path_dataset, 'ONLINE_DHG__all_skeletons.pytorchdata'))
        elif articulations == 'image' and not segment_sequences:
            x_dataset = torch.load(os.path.join(path_dataset, 'ONLINE_DHG__all_skeletons_image.pytorchdata'))
        elif articulations == 'world' and not segment_sequences:
            x_dataset = torch.load(os.path.join(path_dataset, 'ONLINE_DHG__all_skeletons_world.pytorchdata'))
        elif articulations == 'world_enhanced' and not segment_sequences:
            x_dataset = torch.load(os.path.join(path_dataset, 'ONLINE_DHG__all_skeletons_world_enhanced.pytorchdata'))
        elif articulations == 'simple' and segment_sequences:
            x_dataset = torch.load(os.path.join(path_dataset, 'ONLINE_DHG__all_skeletons_segmented.pytorchdata'))
        elif articulations == 'image' and segment_sequences:
            x_dataset = torch.load(os.path.join(path_dataset, 'ONLINE_DHG__all_skeletons_image_segmented.pytorchdata'))
        elif articulations == 'world' and segment_sequences:
            x_dataset = torch.load(os.path.join(path_dataset, 'ONLINE_DHG__all_skeletons_world_segmented.pytorchdata'))
        elif articulations == 'world_enhanced' and segment_sequences:
            x_dataset = torch.load(os.path.join(path_dataset, 'ONLINE_DHG__all_skeletons_world_enhanced_segmented.pytorchdata'))
        else:
            raise Exception('The dataset you asked for does not exist.')

        if use_14:
            y_dataset = all_labels_14
        else:
            y_dataset = all_labels_28

        if return_both_14_and_28 and not segment_sequences:
            return x_dataset, all_labels_14, all_labels_28
        elif not return_both_14_and_28 and not segment_sequences:
            return x_dataset, y_dataset

        elif return_both_14_and_28 and segment_sequences:
            return x_dataset, all_labels_14, all_labels_28, all_start_end_frames
        else:
            return x_dataset, y_dataset, all_start_end_frames


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


def quicker_load_for_online_dhg_dataset(root='/tmp/ODHG2016dataset', root_out='./data/'):
    """
    Online DHG dataset
    """
    n_subjects = 28
    n_seq_records = 14
    n_gestures_by_unsegmented_sequence = 10

    all_skeletons = []
    all_skeletons_image = []
    all_skeletons_world = []
    all_skeletons_world_enhanced = []
    all_skeletons_segmented = []
    all_skeletons_image_segmented = []
    all_skeletons_world_segmented = []
    all_skeletons_world_enhanced_segmented = []
    tmp_labels_14 = []
    tmp_labels_28 = []
    all_labels_14 = []
    all_labels_28 = []
    cut_indexes = []

    for subject in range(1, n_subjects + 1):

        print('==> Subject {}'.format(subject))

        for sequence in range(1, n_seq_records + 1):

            skeletons = file_to_numpy(root + '/subject_{}/sequence_{}/skeletons.txt'.format(subject, sequence))
            skeletons_image = file_to_numpy(
                root + '/subject_{}/sequence_{}/skeletons_image.txt'.format(subject, sequence))
            skeletons_world = file_to_numpy(
                root + '/subject_{}/sequence_{}/skeletons_world.txt'.format(subject, sequence))
            skeletons_world_enhanced = file_to_numpy(
                root + '/subject_{}/sequence_{}/skeletons_world_enhanced.txt'.format(subject, sequence))
            if skeletons_world is not None:
                skeletons = torch.from_numpy(skeletons)
                skeletons_image = torch.from_numpy(skeletons_image)
                skeletons_world = torch.from_numpy(skeletons_world)
                skeletons_world_enhanced = torch.from_numpy(skeletons_world_enhanced)
                all_skeletons.append(skeletons)
                all_skeletons_image.append(skeletons_image)
                all_skeletons_world.append(skeletons_world)
                all_skeletons_world_enhanced.append(skeletons_world_enhanced)

    for subject in range(1, n_subjects + 1):
        infos_sequences_subject = [s.replace('\n', '').split() for s in
                                   open(root + '/subject_{}_infos_sequences.txt'.format(subject), 'r').readlines()]
        start_ends_subject = numpy.array(infos_sequences_subject[2::3])
        cut_indexes.append(start_ends_subject)
    cut_indexes = numpy.vstack([a for a in cut_indexes])
    first_col = numpy.array([0 for a in all_skeletons_world]).T[:, numpy.newaxis]
    last_col = numpy.array([len(a) for a in all_skeletons_world]).T[:, numpy.newaxis] - 1
    cut_indexes = numpy.hstack([first_col, cut_indexes, last_col])
    cut_indexes = cut_indexes.tolist()
    for ase_idx in range(len(cut_indexes)):
        if int(cut_indexes[ase_idx][len(cut_indexes[ase_idx]) - 1]) == int(
                cut_indexes[ase_idx][len(cut_indexes[ase_idx]) - 2]):
            cut_indexes[ase_idx].pop()

    def cut_array(arr, cuts_list):
        all_cuts = []
        for cut_start, cut_end in zip(cuts_list, cuts_list[1:]):
            all_cuts.append(arr[int(cut_start):int(cut_end)])
        return all_cuts

    def quick_flatten(arr):
        return [item for sublist in arr for item in sublist]

    all_skeletons_segmented = quick_flatten(
        [cut_array(all_skeletons[i], cut_indexes[i]) for i in range(len(cut_indexes))])
    all_skeletons_image_segmented = quick_flatten(
        [cut_array(all_skeletons_image[i], cut_indexes[i]) for i in range(len(cut_indexes))])
    all_skeletons_world_segmented = quick_flatten(
        [cut_array(all_skeletons_world[i], cut_indexes[i]) for i in range(len(cut_indexes))])
    all_skeletons_world_enhanced_segmented = quick_flatten(
        [cut_array(all_skeletons_world_enhanced[i], cut_indexes[i]) for i in range(len(cut_indexes))])

    for subject in range(1, n_subjects + 1):
        infos_sequences_subject = [s.replace('\n', '').split() for s in
                                   open(root + '/subject_{}_infos_sequences.txt'.format(subject), 'r').readlines()]
        for i in range(int(len(infos_sequences_subject) / 3)):
            info_seq_i = infos_sequences_subject[3 * i:3 * (i + 1)]
            for g in range(len(info_seq_i[0])):
                tmp_labels_14.append(int(info_seq_i[0][g]))
                tmp_labels_28.append(int(info_seq_i[0][g]) * int(info_seq_i[1][g]))

    tmp_labels_14 = [tmp_labels_14[i * 10:(i + 1) * 10] for i in range(int(len(tmp_labels_14) / 10))]
    tmp_labels_28 = [tmp_labels_28[i * 10:(i + 1) * 10] for i in range(int(len(tmp_labels_28) / 10))]

    for l in range(len(cut_indexes)):
        for i in range(len(cut_indexes[l]) - 1):
            if i % 2 == 0:
                all_labels_14.append(0)
                all_labels_28.append(0)
            else:
                all_labels_14.append(tmp_labels_14[l][int((i - 1) / 2)])
                all_labels_28.append(tmp_labels_28[l][int((i - 1) / 2)])

    print('Saving to disk...')
    torch.save(all_skeletons, root_out + '/ONLINE_DHG__all_skeletons.pytorchdata')
    torch.save(all_skeletons_image, root_out + '/ONLINE_DHG__all_skeletons_image.pytorchdata')
    torch.save(all_skeletons_world, root_out + '/ONLINE_DHG__all_skeletons_world.pytorchdata')
    torch.save(all_skeletons_world_enhanced, root_out + '/ONLINE_DHG__all_skeletons_world_enhanced.pytorchdata')
    torch.save(all_skeletons_segmented, root_out + '/ONLINE_DHG__all_skeletons_segmented.pytorchdata')
    torch.save(all_skeletons_image_segmented, root_out + '/ONLINE_DHG__all_skeletons_image_segmented.pytorchdata')
    torch.save(all_skeletons_world_segmented, root_out + '/ONLINE_DHG__all_skeletons_world_segmented.pytorchdata')
    torch.save(all_skeletons_world_enhanced_segmented,
               root_out + '/ONLINE_DHG__all_skeletons_world_enhanced_segmented.pytorchdata')
    torch.save(all_labels_14, root_out + '/ONLINE_DHG__all_labels_14.pytorchdata')
    torch.save(all_labels_28, root_out + '/ONLINE_DHG__all_labels_28.pytorchdata')
    torch.save(cut_indexes, root_out + '/ONLINE_DHG__all_start_end_frames.pytorchdata')
    print('Saved to disk.')
