import pathlib
import warnings

import pandas
from scipy import random


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
