grid_params = {
    'max_epochs': [2000],
    'batch_size': [64],
    'lr': [1e-4],
    'module__dropout': [0.4],
    'module__activation_fct': ['prelu'],
    'module__preprocess': [None],
    'module__conv_type': ['regular'],

    'module__channel_list': [
        # Change size of the conv layers and number of layers


        # [(22, 3)],
        # [(22, 3), (22, 3)],
        # [(22, 3), (22, 3), (22, 3)],
        # [(22, 3), (22, 3), (22, 3), (22, 3)],
        
        # [(66, 3)],
        # [(66, 3), (66, 3)],
        [(66, 3), (66, 3), (66, 3)],
        [(96, 1)],

        # [(66, 3), (66, 3), (66, 3), (66, 3)],

        # [(132, 3)],
        # [(132, 3), (132, 3)],
        # [(132, 3), (132, 3), (132, 3)],
        # [(132, 3), (132, 3), (132, 3), (132, 3)],

        # #


        # [(66, 66), (66, 66), (66, 66)],
        # [(66, 1), (66, 1), (66, 1)],
        # [(66, 22), (66, 22), (66, 22)],
    ],
    # if preprocess: list of tuples [<(C_preprocess, None)>, (C_conv1, G_conv1), (C_conv2, G_conv2), (C_conv3, G_conv3), ...]
    # [(66, None), (66, 33), (66, 11)],
    # [(66, None), (66, 66), (66, 11)],
    # else: list of tuples [<(C_preprocess, None)>, (C_conv1, G_conv1), (C_conv2, G_conv2), (C_conv3, G_conv3), ...]
    # [(66,33), (66,11)],

    'module__fc_hidden_layers': [
        [2048, 128],
    ],

    'module__temporal_attention': [None],  # can also be 'dot_attention', 'general_attention'
}
