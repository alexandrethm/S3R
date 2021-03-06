# See the effects of the groups parameter

grid_params = {
    'max_epochs': [2000],
    'batch_size': [128],
    'lr': [1e-4],
    'module__dropout': [0.4],
    'module__activation_fct': ['prelu'],
    'module__preprocess': ['LSC'],
    'module__conv_type': ['regular'],

    'module__channel_list': [
        [(22, None), (22, 1)],
        [(44, None), (22, 1)],
        [(66, None), (22, 1)],
        [(88, None), (22, 1)],
        [(110, None), (22, 1)],
        [(132, None), (22, 1)],
],
    # if preprocess: list of tuples [<(C_preprocess, None)>, (C_conv1, G_conv1), (C_conv2, G_conv2), (C_conv3, G_conv3), ...]
    # [(66, None), (66, 33), (66, 11)],
    # [(66, None), (66, 66), (66, 11)],
    # else: list of tuples [<(C_preprocess, None)>, (C_conv1, G_conv1), (C_conv2, G_conv2), (C_conv3, G_conv3), ...]
    # [(66,33), (66,11)],

    'module__fc_hidden_layers': [
        [1024, 128],
    ],

    'module__temporal_attention': [None],  # can also be 'dot_attention', 'general_attention'
}