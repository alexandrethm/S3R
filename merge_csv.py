import ast
import glob
import math
import numpy
import os
import pandas


def list_csv_in_folder(path='results/grid_search_1122_1128'):
    # note: it is important to sort the filenames
    csv_files = sorted(glob.glob(os.path.join(path, '*.csv')))
    csvs = [pandas.read_csv(c).values for c in csv_files]
    return csv_files, csvs


def get_columns(csv_matrix):
    return csv_matrix[:, 0].tolist()


def get_value_for_key(csv_matrix, key):
    if key not in csv_matrix[:, 0].tolist():
        raise AttributeError('Key not present in CSV')
    return csv_matrix[csv_matrix[:, 0].tolist().index(key)][1]


def is_same_experiment(csv_file_a, csv_file_b):
    if sorted(get_columns(csv_file_a)) != sorted(get_columns(csv_file_b)):
        return False
    if get_columns(csv_file_a) != get_columns(csv_file_b):
        raise Warning(
            'The keys in the CSV files are not ordered correctly. The program will likely not work correctly.')

    for k in get_columns(csv_file_a):
        ka = get_value_for_key(csv_file_a, k)
        kb = get_value_for_key(csv_file_b, k)
        if ka != kb:
            # return false, except when the two fields are NaN (i.e. blank/missing)
            if isinstance(ka, float) and isinstance(kb, float):
                if math.isnan(ka) and math.isnan(kb):
                    pass
            else:
                # print('key {} is different: {} /// {}'.format(k, ka, kb))
                return False
    return True


def merge_and_write_csvs(csv_list, name):
    if len(csv_list) == 0:
        pass
    if len(csv_list) == 1:
        return csv_list[0]

    def _get_ki(arr, k):
        ki = arr[get_columns(arr).index(k)][2]
        ki = ast.literal_eval(ki)
        ki = numpy.array(ki, dtype=float)
        return ki

    def _pad_array(A, size, with_last_value=True):
        """
        >>> _pad_array([1,2,3], 8)
        # [1 2 3 0 0 0 0 0]
        """
        t = size - len(A)
        fillval = A[-1] if with_last_value else 0
        return numpy.pad(A, pad_width=(0, t), mode='constant', constant_values=(fillval))

    def _vstack_any_length(arr_list):
        m = max([len(a) for a in arr_list])
        return numpy.vstack([_pad_array(a, m) for a in arr_list])

    keys_to_merge = ['train_loss', 'valid_acc', 'valid_loss']

    # the first keys are common
    csv_out = csv_list[0][:-len(keys_to_merge)]
    csv_out = csv_out.T
    csv_out = csv_out[:-1]

    recorded_metrics = []

    for k in keys_to_merge:
        # individual keys
        for i in range(len(csv_list)):
            ki = _get_ki(csv_list[i], k)
            arr = numpy.hstack([k + '_' + str(i), ki])
            recorded_metrics.append(arr)
        # min/max/mu/sigma keys
        cat = _vstack_any_length([_get_ki(csv_list[i], k) for i in range(len(csv_list))])
        kmin = cat.min(axis=0)
        kmax = cat.max(axis=0)
        kmean = cat.mean(axis=0)
        kstd = cat.std(axis=0)
        recorded_metrics.append(numpy.hstack([k + '_min', kmin]))
        recorded_metrics.append(numpy.hstack([k + '_max', kmax]))
        recorded_metrics.append(numpy.hstack([k + '_mean', kmean]))
        recorded_metrics.append(numpy.hstack([k + '_std', kstd]))

    recorded_metrics = _vstack_any_length(recorded_metrics).T

    df_csv_out = pandas.DataFrame(data=csv_out)
    df_recorded_metrics = pandas.DataFrame(data=recorded_metrics)
    df_out = pandas.concat([df_csv_out, df_recorded_metrics], axis=1, sort=False)

    # Save as CSV file
    df_out.to_csv(path_or_buf=name.replace('data_', 'experiment_'))


def main():
    csv_files, csvs = list_csv_in_folder()  # todo: support path

    experiment_start_idx = 0
    experiment_list = [csvs[0]]

    res_merged = []

    for i in range(len(csvs) - 1):
        if is_same_experiment(csvs[i], csvs[i + 1]):
            experiment_list.append(csvs[i + 1])
        else:
            # end of a experiment. merge them
            # then free the list / idx for the next experiment
            merged = merge_and_write_csvs(experiment_list, name=csv_files[experiment_start_idx])
            res_merged.append(merged)
            experiment_start_idx = i
            experiment_list = [csvs[i + 1]]

    # finally check if there are remaining items in the list
    # and merge them
    if len(res_merged) > 0:
        merged = merge_and_write_csvs(experiment_list, name=csv_files[experiment_start_idx])
        res_merged.append(merged)

    return res_merged


if __name__ == '__main__':
    main()
