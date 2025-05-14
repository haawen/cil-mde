"""
Run the MSPN refinement after getting the guidance features
"""
from argparse import Namespace
import os
import random
import torch
import numpy as np
from matplotlib import pyplot as plt
from run_guidance import load_sample
from lib.model.MSPN import MSPN
from lib.model.backbone import Backbone

args = Namespace(
    data_name='NYU',
    mode='SDR',
    embed_dim=64,
    pretrain='test_models/SDR_NYU.pt',
    prop_time=6,
)
# Add your own paths to these directories
RGB_DIR = ''
DEPTH_MD_DIR = RGB_DIR
TOTAL_VARIANCE_DIR = ''
SAMPLE_NUM = random.choice(os.listdir(TOTAL_VARIANCE_DIR))[7:13]


def depth_and_mask_from_var(total_var, depth_md, threshold=0.1):
    depth = torch.zeros_like(depth_md)
    depth[total_var < threshold] = depth_md[total_var < threshold]
    return depth, (total_var < threshold) * 1.0


def depth_and_mask_from_rand(depth_md, filter_num=1):
    init_depth = depth_md.squeeze(0).numpy()
    mask = np.random.rand(*init_depth.shape)
    if filter_num < 1:
        mask = mask < filter_num  # Keep % of the depth values
    else:
        new_mask = np.zeros_like(mask)
        for _ in range(filter_num):
            w_idx = random.randrange(0, mask.shape[1])
            h_idx = random.randrange(0, mask.shape[2])
            new_mask[:, w_idx, h_idx] = mask[:, w_idx, h_idx]
        mask = new_mask > 0.0
    sparse_depth = np.zeros_like(init_depth)
    sparse_depth[mask] = init_depth[mask]  # Retain only the sampled values

    depth = torch.tensor(sparse_depth).unsqueeze(0)  # Shape: (1, 1, H, W)
    mask = torch.tensor(mask).unsqueeze(0).float()
    print(mask.type())
    return depth, mask


def get_num_sample_tensor(mask):
    return torch.Tensor([len(mask.nonzero())])


def get_mask_from_var(total_var, threshold=0.1):
    return (total_var < threshold) * 1.0


def prepare_inputs(rgb_path, var_path, depth_md_path, device):
    rgb, variance, depth_md = load_sample(rgb_path, var_path, depth_md_path)
    #depth, mask_init = depth_and_mask_from_rand(depth_md)
    depth, mask_init = depth_and_mask_from_var(variance, depth_md)
    num_sample = get_num_sample_tensor(mask_init.nonzero())

    rgb = rgb.to(device)
    depth = depth.to(device)
    depth_md = depth_md.to(device)
    mask_init = mask_init.to(device)

    num_sample = num_sample.to(device)

    guidance_net = Backbone(args).to(device)
    _, guide = guidance_net(rgb=rgb, depth=depth, depth_MD=depth_md)

    mask = mask_init
    pred_init = depth_md * (1 - mask) + depth * mask

    y_inter = [pred_init, ]
    mask_inter = [mask_init, ]

    return pred_init, y_inter, mask_inter, guide, depth, mask_init, num_sample


def load_model(model):
    checkpoint = torch.load(args.pretrain)
    _, _ = model.load_state_dict(checkpoint['net'], strict=False)

    print(f'Checkpoint loaded from {args.pretrain}!')


def visualize_output(output):
    mask_inter = output['mask_inter']
    y_inter = output['pred_inter']
    fig = plt.figure(figsize=(10, 10))
    plt.tight_layout()
    for i, yy in enumerate([mask_inter[0], mask_inter[-1], y_inter[0], y_inter[-1]]):
        fig.add_subplot(2, 2, i+1)
        plt.imshow(yy.squeeze(0).squeeze(0).cpu().numpy())
    plt.show()


def get_model_output(model, pred_init, y_inter, mask_inter, guide, depth, mask_init, num_sample):
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
    return output


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    model = MSPN(args).to(device)
    model.eval()

    load_model(model)

    # RGB image
    rgb_file = os.path.join(RGB_DIR, f'sample_{SAMPLE_NUM}_rgb.png')
    # total variance map
    var_file = os.path.join(TOTAL_VARIANCE_DIR, f'sample_{SAMPLE_NUM}_depth_var_total.npy')
    # init depth estimation
    depth_md_file = os.path.join(DEPTH_MD_DIR, f'sample_{SAMPLE_NUM}_depth.npy')

    pred_init, y_inter, mask_inter, guide, depth, mask_init, num_sample = prepare_inputs(rgb_file, var_file, depth_md_file, device)

    output = get_model_output(model, pred_init, y_inter, mask_inter, guide, depth, mask_init, num_sample)

    visualize_output(output)


if __name__ == '__main__':
    main()
