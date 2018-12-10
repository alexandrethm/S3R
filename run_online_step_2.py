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

# keep quiet, scipy
utils.hide_scipy_zoom_warnings()

# comet ml
os.environ['COMET_ML_API_KEY'] = 'Tz0dKZfqyBRMdGZe68FxU3wvZ'

# output folder
now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
if not os.path.exists('./results/' + now):
    os.makedirs('./results/' + now)


# -------------
# Data
# -------------
# Load the dataset
batch_size = 64
window_length = 100

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
model = torch.load('results/model.cudapytorchmodel')
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
    all_outs = []
    for t in range(x_seq.shape[0]):
        if t < x_seq.shape[0] - window_length:
            out_t = model(x_seq[t:t + window_length].unsqueeze(0))
            all_outs.append(out_t.cpu().detach())
    all_a = numpy.array(
        [[softmax(prob).squeeze().numpy()[i] for prob in all_outs] for i in range(15)]
    )
    segmented = numpy.array([numpy.int8(all_a[:, t] == all_a[:, t].max()) for t in range(all_a.shape[1])]).T
    y_ground_truth = y_test_list[idx].cpu().numpy().T

    # -------------
    # Figure
    # -------------
    f = plt.figure()
    # --- predicted: segmentation ---
    # for i in range(15):
    #     plt.plot(range(len(all_outs)), segmented[i])
    # --- predicted: probabilities ---
    for i in range(15):
        plt.plot(range(len(all_outs)), all_a[i])
        plt.title('Probas gestes sequence #{}'.format(idx + 1))
    # plt.show()
    f.savefig("./results/" + now + "/Sequence_{}_predicted.pdf".format(idx + 1), bbox_inches='tight')
    print('Sequence {} -- Produced figure'.format(idx + 1))

    # -------------
    # CSV
    # -------------
    pandas.DataFrame(data=all_a.T).to_csv("./results/" + now + "/Sequence_{}_predicted.csv".format(idx + 1),
                                          header=True, index=False)
    pandas.DataFrame(data=segmented.T).to_csv(
        "./results/" + now + "/Sequence_{}_predicted_segmented.csv".format(idx + 1), header=True, index=False)
    pandas.DataFrame(data=y_ground_truth.T).to_csv("./results/" + now + "/Sequence_{}_ground_truth.csv".format(idx + 1),
                                                   header=True, index=False)
    print('Sequence {} -- Produced CSV files'.format(idx + 1))

    # -------------
    # Junk (to delete)
    # -------------
    # pandas.DataFrame(data=all_a.T).to_json("./results/Y_gestes_sequence_{}.json".format(idx+1), index=False)
    # numpy.savetxt("./results/Y_gestes_sequence_{}.csv".format(idx+1), all_a.T, delimiter=",")
    # import json
    # chart_data = pandas.DataFrame(data=all_a).to_dict(orient='series')
    # chart_data = json.dumps(chart_data, indent=2)
    # data = {'chart_data': chart_data}
    # print(data)
    # print(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    # plt.rcParams["figure.figsize"] = [16, 9]
    # flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
    # seaborn.set_palette(flatui)
    # seaborn.palplot(seaborn.color_palette())
