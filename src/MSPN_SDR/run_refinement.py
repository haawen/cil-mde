"""
Run the MSPN refinement after getting the guidance features
"""

from argparse import Namespace
from lib.model.MSPN import MSPN
args = Namespace(
    dir_data='dataset/nyudepthv2_SDR/',
    data_name='NYU',
    split_json='data_json/nyu.json',
    patch_height=228,
    patch_width=304,
    top_crop=0,
    seed=43,
    gpus="0",
    port='29500',
    address='localhost',
    num_threads=4,
    no_multiprocessing=False,
    prop_time=6,
    loss='1.0*L1+1.0*L2',
    epochs=72,
    milestones=[36, 48, 56, 64],
    opt_level='O0',
    resume=False,
    test_only=True,
    batch_size=2,
    max_depth=10.0,
    augment=True,
    no_augment=True,
    lidar_lines=64,
    test_crop=False,
    num_summary=4,
    lr=0.001,
    gamma=0.5,
    optimizer='ADAMW',
    momentum=0.9,
    betas=(0.9, 0.999),
    epsilon=1e-8,
    weight_decay=0.01,
    warm_up=False,
    no_warm_up=True,
    log_dir='experiments/',
    print_freq=1,
    save_full=True,
    save_image=True,
    save_result_only=False,
    save_result_npy=False,
    mode='SDR',
    embed_dim=64,
    num_sparse_depth_train=500,
    num_sparse_depth_test=500,
    train_with_random_sds=True,
    pretrain='test_models/SDR_NYU.pt'
)

model = MSPN(args)
