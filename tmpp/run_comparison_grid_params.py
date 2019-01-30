{
    'max_epochs': 2000,
    'batch_size': 64,
    'lr': 10e-4,
    'module__dropout': 0.4,
    'module__activation_fct': 'prelu',
    'module__preprocess': None,
    'module__conv_type': 'regular',

    # relation for 22 joints ? 132=6*22
    'module__channel_list': [(132, 11), (22, 11)],

    # relation for 14 output gestures ? 70=5*14, 1400=100*14
    'module__fc_hidden_layers': [1400, 70],

    'module__temporal_attention': None,
}