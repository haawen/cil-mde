"""
Run the MSPN refinement after getting the guidance features
"""
import os
from argparse import Namespace
from run_guidance import load_sample
from lib.model.MSPN import MSPN
import torch
import numpy as np



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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

model = MSPN(args).to(device)
model.eval()

checkpoint = torch.load(args.pretrain)
key_m, key_u = model.load_state_dict(checkpoint['net'], strict=False)

print(f'Checkpoint loaded from {args.pretrain}!')


rgb_path = os.path.join("test_images", "test_000000_rgb.png")  # RGB image
depth_path = os.path.join("da2_pipe_preds_npy", "test_000000_depth.npy") # sparse depth map, for now not used
depth_md_path = os.path.join("da2_pipe_preds_npy", "test_000000_depth.npy")  # init depth estimation
guide_path = os.path.join("output_guide.npy")

rgb, depth, depth_md = load_sample(rgb_path, depth_path, depth_md_path)
mask_init = (depth > 0) * 1.0
guide = torch.Tensor(np.load(guide_path)).to(device)
num_sample = torch.Tensor([len(mask_init.nonzero())])

rgb = rgb.to(device)
depth = depth.to(device)
depth_md = depth_md.to(device)
mask_init = mask_init.to(device)
guide = guide.to(device)
num_sample = num_sample.to(device)
#print(guide)

mask = (depth > 0) * 1.0
pred_init = depth_md * (1 - mask) + depth * mask

# Diffusion
y_inter = [pred_init, ]
mask_inter = [mask_init, ]
if args.prop_time > 0:
    with torch.no_grad():
        y, y_inter, mask_inter = model(
            pred_init,
            y_inter,
            mask_inter,
            guide,
            depth,
            mask_init,
            num_sample
        )
else:
    y = pred_init
# Remove negative depth
y = torch.clamp(y, min=0)
# best at first
y_inter.reverse()
mask_inter.reverse()
output = {'pred': y, 'pred_init': pred_init, 'pred_inter': y_inter, 'mask_inter': mask_inter,
          'guidance': guide, 'num_sample': num_sample}

from matplotlib import pyplot as plt
fig = plt.figure(figsize=(10, 10))
for i, yy in enumerate(mask_inter):
    fig.add_subplot(4, 4, i+1)
    plt.imshow(yy.squeeze(0).squeeze(0).cpu().numpy())
plt.show()

fig = plt.figure(figsize=(10, 10))
for i, yy in enumerate(y_inter):
    fig.add_subplot(4, 4, i+1)
    plt.imshow(yy.squeeze(0).squeeze(0).cpu().numpy())
plt.show()