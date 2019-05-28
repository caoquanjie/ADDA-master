class Config(object):
    original_size = 32
    num_channels = 3
    batch_size = 128
    epoch = 2000

    # learning rate
    sgd_learning_rate = 2.0*1e-1
    adam_learning_rate = 2.0*1e-4

    drop_rate = 0.5

    num_step = 400000
    save_period = 800

    depth = 3
    num_classes = 10
    step = 100000



    save_dir = './model'
    summary_dir = './step1/logs'
    step2_summary_dir = './step2/logs'
    save_model_per_step = 2000