class Config(object):
    original_size = 32
    num_channels = 3
    batch_size = 128

    sgd_learning_rate = 2.0*1e-1
    adam_learning_rate = 2.0*1e-4

    drop_rate = 0.5

    summary_dir = './step1/logs'
    save_dir = './model'
    num_step = 400000
    save_period = 800
    phase = 'train'
    is_train = True
    load_model_step = 42


    win_size = 8
    bandwidth = win_size**2
    #batch_size = 32
    eval_batch_size = 50
    loc_std = 0.22

    #num_channels = 1
    depth = 3
    sensor_size = win_size**2 * depth
    minRadius = 8
    hg_size = hl_size = 128
    g_size = 256
    cell_output_size = 256
    loc_dim = 2
    cell_size = 256
    cell_out_size = cell_size
    num_glimpses = 6
    num_classes = 10
    max_grad_norm = 5.

    step = 100000
    lr_start = 1e-3
    lr_min = 1e-4

    #save_dir = './model'
    #summary_dir = './logs'
    step2_summary_dir = './step2/logs'
    save_model_per_step = 2000

    # Monte Carlo sampling
    M = 1
    epoch = 2000