r"""
Usage:
  run_online_step_2.py [--window_length=<window_length>] [--smoothing_window=<smoothing_window>]
  run_online_step_2.py -h | --help

Options:
  -h --help                        Show this screen.
  --window_length=<window_length>  Window length [default: 100].
  --smoothing_window=<smoothing_window>        Smoothing_window size [default: 11].

"""
from docopt import docopt
import comet_ml
import datetime
import os
import matplotlib.pyplot as plt
import numpy
import pandas
import code_S3R.utils.other_utils as utils
import torch
from torch.nn.functional import softmax
from code_S3R import my_nets
from code_S3R.utils.data_utils import OnlineDHGDataset, load_unsequenced_test_dataset
from code_S3R.utils.signal_utils import smooth

# keep quiet, scipy
utils.hide_scipy_zoom_warnings()

# check cuda compatibility
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# comet ml
os.environ['COMET_ML_API_KEY'] = 'Tz0dKZfqyBRMdGZe68FxU3wvZ'

# -------------
# Data
# -------------
# Load the dataset
arguments = docopt(__doc__, version='0.1.1rc')
window_length = int(arguments['--window_length'])
smoothing_window = int(arguments['--smoothing_window'])
balanced = False

# output folder
now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
if balanced:
    now += '_balanced'
else:
    now += '_unbalanced'
now += "_window_{}_smoothing_{}".format(window_length, smoothing_window)
if not os.path.exists('./results/' + now):
    os.makedirs('./results/' + now)

x_test_list, y_test_list = load_unsequenced_test_dataset()
x_test_list = [x.cuda() if device == 'cuda' else x for x in x_test_list]
y_test_list = [y.cuda() if device == 'cuda' else y for y in y_test_list]
y_test_list = [y[window_length:] for y in y_test_list]

# -------------
# Model
# -------------
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
if not balanced:
    print('INFO: Loading pretrained model trained on the full _unbalanced_ training set.')
    model = torch.load('results/model__window_{}.cudapytorchmodel'.format(window_length),
                       map_location=lambda storage, loc: storage)
else:
    print('INFO: Loading pretrained model trained on the full _balanced_ sampled training set.')
    model = torch.load('results/model__balanced__window_{}.cudapytorchmodel'.format(window_length),
                       map_location=lambda storage, loc: storage)
if device is 'cuda':
    model = model.cuda(device=device)
model.eval()
print('Loaded pretrained model')

# -------------
# For each sequence...
# -------------
for idx, x_seq in enumerate(x_test_list):
    # -------------
    # Prediction / Segmentation / Ground truth
    # -------------
    # Prediction (not balanced), without softmax
    all_outs = []
    for t in range(x_seq.shape[0]):
        if t < x_seq.shape[0] - window_length:
            out_t = model(x_seq[t:t + window_length].unsqueeze(0))
            all_outs.append(out_t.cpu().detach())

    # Prediction (not balanced)
    predicted = numpy.array(
        [[softmax(prob_at_time_t).squeeze().numpy()[i] for prob_at_time_t in all_outs] for i in range(15)]
    )

    # Segmentation
    segmented = numpy.array([numpy.int8(predicted[:, t] == predicted[:, t].max()) for t in range(predicted.shape[1])]).T

    # Ground truth
    y_ground_truth = y_test_list[idx].cpu().numpy().T

    # ------ Everything below (until Figure) is more or less a work in progress ------

    # Calculate the frequency of each gesture in the training set
    class_sample_counts = torch.from_numpy(numpy.array([d[1].numpy() for d in OnlineDHGDataset(set='train')])).sum(
        dim=0)
    weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)

    # Bonus/Test: ponderation
    ponderated = numpy.array(
        [[softmax(prob_at_time_t * weights).squeeze().numpy()[i] for prob_at_time_t in all_outs] for i in range(15)]
    )
    segmented_ponderated = numpy.array(
        [numpy.int8(ponderated[:, t] == ponderated[:, t].max()) for t in
         range(ponderated.shape[1])]).T

    # Bonus/Test: smoothing
    smoothed = numpy.array([smooth(predicted[p], window_len=smoothing_window, window='flat')
                            for p
                            in range(15)])[:, int((smoothing_window - 1) / 2):-int((smoothing_window - 1) / 2)]
    segmented_smoothed = numpy.array(
        [numpy.int8(smoothed[:, t] == smoothed[:, t].max()) for t in range(smoothed.shape[1])]).T

    # -------------
    # Figure
    # -------------
    if True:
        f = plt.figure()
        # --- predicted: segmentation ---
        # for i in range(15):
        #     plt.plot(range(len(all_outs)), segmented[i])
        # --- predicted: probabilities ---
        for i in range(15):
            plt.plot(range(len(all_outs)), predicted[i])
            plt.title('Probas gestes sequence #{}'.format(idx + 1))
        # plt.show()
        f.savefig("./results/" + now + "/Sequence_{}_predicted.pdf".format(idx + 1), bbox_inches='tight')
        del f
        print('Sequence {} -- Produced figure'.format(idx + 1))

    # -------------
    # CSV
    # -------------
    if True:
        # --- prediction / segmentation / ground truth ---
        pandas.DataFrame(data=predicted.T).to_csv("./results/" + now + "/Sequence_{}_predicted.csv".format(idx + 1),
                                                  header=True, index=False)
        pandas.DataFrame(data=segmented.T).to_csv(
            "./results/" + now + "/Sequence_{}_segmented.csv".format(idx + 1), header=True, index=False)
        pandas.DataFrame(data=y_ground_truth.T).to_csv(
            "./results/" + now + "/Sequence_{}_ground_truth.csv".format(idx + 1),
            header=True, index=False)

        # --- bonus ---
        pandas.DataFrame(data=smoothed.T).to_csv(
            "./results/" + now + "/Sequence_{}_smoothed_{}.csv".format(idx + 1, smoothing_window),
            header=True, index=False)
        pandas.DataFrame(data=ponderated.T).to_csv(
            "./results/" + now + "/Sequence_{}_ponderated.csv".format(idx + 1), header=True, index=False)

        pandas.DataFrame(data=segmented_smoothed.T).to_csv(
            "./results/" + now + "/Sequence_{}_segmented_smoothed_{}.csv".format(idx + 1, smoothing_window),
            header=True, index=False)
        pandas.DataFrame(data=segmented_ponderated.T).to_csv(
            "./results/" + now + "/Sequence_{}_segmented_ponderated_{}.csv".format(idx + 1, smoothing_window),
            header=True, index=False)

        print('Sequence {} -- Produced CSV files'.format(idx + 1))

        # -------------
        # Junk (to delete)
        # -------------
        # pandas.DataFrame(data=predicted.T).to_json("./results/Y_gestes_sequence_{}.json".format(idx+1), index=False)
        # numpy.savetxt("./results/Y_gestes_sequence_{}.csv".format(idx+1), predicted.T, delimiter=",")
        # import json
        # chart_data = pandas.DataFrame(data=predicted).to_dict(orient='series')
        # chart_data = json.dumps(chart_data, indent=2)
        # data = {'chart_data': chart_data}
        # print(data)
        # print(plt.rcParams['axes.prop_cycle'].by_key()['color'])
        # plt.rcParams["figure.figsize"] = [16, 9]
        # flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
        # seaborn.set_palette(flatui)
        # seaborn.palplot(seaborn.color_palette())
