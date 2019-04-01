# See the effects of the groups parameter

grid_params = {
    'max_epochs': [2000],
    'batch_size': [128],
    'lr': [1e-4],
    'module__dropout': [0.4],
    'module__activation_fct': ['prelu'],
    'module__preprocess': [None],
    'module__conv_type': ['regular'],

    'module__channel_list': [
        # groups for small and shallow networks ?
        [(22, 1)],
        [(22, 2)],
        [(22, 11)],
        [(22, 22)],

        # groups for bigger and deeper networks ?
        [(96, 1), (96, 1)],
        [(96, 2), (96, 2)],
        [(96, 3), (96, 3)],
        [(96, 6), (96, 6)],
        
        [(96, 1), (96, 1), (96, 1), (96, 1)],
        [(96, 2), (96, 2), (96, 2), (96, 2)],
        [(96, 3), (96, 3), (96, 3), (96, 3)],
        [(96, 6), (96, 6), (96, 6), (96, 6)],
        
        # a lot of values of groups ?
        [(66, 1)],
        [(66, 2)],
        [(66, 3)],
        [(66, 6)],
        [(66, 11)],
        [(66, 22)],
        [(66, 33)],
        [(66, 66)],
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