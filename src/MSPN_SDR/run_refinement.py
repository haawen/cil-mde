"""
Run the MSPN refinement after getting the guidance features
"""
from argparse import Namespace
import os
import random
import torch
import numpy as np
from matplotlib import pyplot as plt
from .run_guidance import load_sample, Backbone
from .lib.model.MSPN import MSPN

args = Namespace(
    data_name='NYU',
    mode='SDR',
    embed_dim=64,
    pretrain='test_models/SDR_NYU.pt',
    prop_time=6,
)
# Add your own paths to these directories
RGB_DIR = os.path.join('/')
DEPTH_MD_DIR = os.path.join('/')
TOTAL_VARIANCE_DIR = os.path.join('/')


def sample_depth_from_var(total_var, depth_md, threshold=0.01):
    depth = torch.zeros_like(depth_md)
    cutoff = total_var.min() + (total_var.max() - total_var.min()) * threshold
    depth[total_var < cutoff] = depth_md[total_var < cutoff]
    return depth


def sample_depth_from_rand(depth_md, filter_num=1):
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


def get_num_sample_tensor(depth):
    return torch.Tensor([len(depth.nonzero())])


def get_mask_from_var(total_var, threshold=0.01):
    return (total_var < threshold) * 1.0


def load_inputs(rgb_path, var_path, depth_md_path, threshold=0.01):
    rgb, variance, depth_md = load_sample(rgb_path, var_path, depth_md_path)
    #depth, mask_init = depth_and_mask_from_rand(depth_md)
    depth = sample_depth_from_var(variance, depth_md, threshold=threshold)
    num_sample = get_num_sample_tensor(depth)

    return rgb, depth, depth_md, variance, num_sample

def prepare_inputs(rgb, depth, depth_md, variance, num_sample, guidance_net, device):
    rgb = rgb.to(device)
    depth = depth.to(device)
    depth_md = depth_md.to(device)
    variance = variance.to(device)

    num_sample = num_sample.to(device)

    with torch.no_grad():
        _, guide = guidance_net(rgb=rgb, depth=depth, depth_MD=depth_md)

    y_inter = [depth_md, ]
    var_inter = [variance, ]

    return depth_md, guide, depth, variance, num_sample, y_inter, var_inter


def load_model(model, args):
    checkpoint = torch.load(args.pretrain)
    _, _ = model.load_state_dict(checkpoint['net'], strict=False)

    print(f'Checkpoint loaded from {args.pretrain}!')


def visualize_output(output, sample_num):
    var_inter = output['var_inter']
    y_inter = output['pred_inter']

    gt = np.load(os.path.join(RGB_DIR, f'sample_{sample_num}_depth.npy'))
    gt = torch.tensor(gt).unsqueeze(0)

    def get_from_func(ll, py_func, torch_func):
        return py_func([torch_func(l) for l in ll])

    var_min = get_from_func(var_inter, min, torch.min)
    var_max = get_from_func(var_inter, max, torch.max)

    depth_min = min(get_from_func(y_inter, min, torch.min), gt.min())
    depth_max = max(get_from_func(y_inter, max, torch.max), gt.max())

    #fig = plt.figure()
    plt.tight_layout()
    print("Variances:")
    for i, yy in enumerate(var_inter):
        #fig.add_subplot(5, len(var_inter)//2 + 1, i+1)
        plt.imshow(yy.squeeze(0).squeeze(0).cpu().numpy())#, vmin=var_min, vmax=var_max)
        plt.show()
    print("Depths:")
    for i, yy in enumerate(y_inter):
        #fig.add_subplot(5, len(y_inter)//2 + 1, i + len(y_inter) + 2)
        plt.imshow(yy.squeeze(0).squeeze(0).cpu().numpy())#, vmin=depth_min, vmax=depth_max)
        plt.show()
    #fig.add_subplot(5, len(var_inter)//2 + 1, 4 * (len(var_inter)//2 + 1) + 1)
    print("Ground Truth:")
    plt.imshow(gt.squeeze(0).squeeze(0).cpu().numpy())#, vmin=depth_min, vmax=depth_max)
    plt.show()


def get_model_output(model, pred_init, y_inter, var_inter, guide, depth, var_init, num_sample):
    if args.prop_time > 0:
        #with torch.no_grad():
        y, y_inter, var_inter = model(
            pred_init,
            y_inter,
            var_inter,
            guide,
            depth,
            var_init,
            num_sample
        )
    else:
        y = pred_init
    # Remove negative depth
    y = torch.clamp(y, min=0)
    # best at first
    #y_inter.reverse()
    #var_inter.reverse()
    output = {'pred': y, 'pred_init': pred_init, 'pred_inter': y_inter, 'var_inter': var_inter,
            'guidance': guide, 'num_sample': num_sample}
    return output

def eval_depth(pred, target):
    assert pred.shape == target.shape

    thresh = torch.max((target / pred), (pred / target))

    d1 = torch.sum(thresh < 1.25).float() / len(thresh)

    diff = pred - target
    diff_log = torch.log(pred) - torch.log(target)

    abs_rel = torch.mean(torch.abs(diff) / target)

    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    mae = torch.mean(torch.abs(diff))

    silog = torch.sqrt(torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2))

    return {'d1': d1.detach().item(), 'abs_rel': abs_rel.detach().item(),'rmse': rmse.detach().item(), 'mae': mae.detach().item(), 'silog':silog.detach().item()}

def compute_ause(errors, uncertainties, num_points=100):
    n = len(errors)
    if n == 0:
        return 0.0

    sorted_indices_uncertainty = np.argsort(uncertainties)
    sorted_errors_uncertainty = errors[sorted_indices_uncertainty]

    sorted_indices_error = np.argsort(errors)
    sorted_errors_error = errors[sorted_indices_error]

    percentiles_removed = np.linspace(0, 100, num_points)
    
    model_curve = []
    oracle_curve = []
    
    for p_removed in percentiles_removed:
        k = int(n * (p_removed / 100))
        m = n - k # number of points kept

        if m > 0:
            model_error = np.mean(sorted_errors_uncertainty[:m])
            oracle_error = np.mean(sorted_errors_error[:m])
            model_curve.append(model_error)
            oracle_curve.append(oracle_error)
        else: # m == 0, 100% removed
            model_curve.append(0.0)
            oracle_curve.append(0.0)

    ause = np.trapz(np.array(model_curve) - np.array(oracle_curve), x=percentiles_removed) / 100.0

    return ause


def compute_aurg(errors, uncertainties, num_points=100):
    n = len(errors)
    if n == 0:
        return 0.0

    baseline_error = np.mean(errors)

    sorted_indices = np.argsort(uncertainties)
    sorted_errors = errors[sorted_indices]

    percentiles_removed = np.linspace(0, 100, num_points)
    risk_gains = []

    for p_removed in percentiles_removed:
        k = int(n * (p_removed / 100))
        m = n - k

        if m > 0:
            error_after_removal = np.mean(sorted_errors[:m])
            risk_gain = baseline_error - error_after_removal
            risk_gains.append(risk_gain)
        else:
            risk_gains.append(0.0)

    aurg = np.trapz(risk_gains, x=percentiles_removed) / 100.0
    return aurg

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    model = MSPN(args).to(device)
    model.eval()

    load_model(model)

    SAMPLE_NUM = random.choice(os.listdir(TOTAL_VARIANCE_DIR))[7:13]
    SAMPLE_NUM = '009067'

    # RGB image
    rgb_file = os.path.join(RGB_DIR, f'sample_{SAMPLE_NUM}_rgb.png')
    # total variance map
    var_file = os.path.join(TOTAL_VARIANCE_DIR, f'sample_{SAMPLE_NUM}_depth_var_total.npy')
    # init depth estimation
    depth_md_file = os.path.join(DEPTH_MD_DIR, f'sample_{SAMPLE_NUM}_depth_mean.npy')

    guidance_net = Backbone(args).to(device)

    rgb, depth, depth_md, variance, num_sample = load_inputs(rgb_file, var_file, depth_md_file)

    pred_init, y_inter, var_inter, guide, depth, var_init, num_sample = prepare_inputs(rgb, depth, depth_md, variance, num_sample, guidance_net, device)

    output = get_model_output(model, pred_init, y_inter, var_inter, guide, depth, var_init, num_sample)

    visualize_output(output, SAMPLE_NUM)

    gt = np.load(os.path.join(RGB_DIR, f'sample_{SAMPLE_NUM}_depth.npy'))
    gt = torch.tensor(gt).unsqueeze(0).unsqueeze(0).to(device)

    print(eval_depth(pred_init, gt))
    print(eval_depth(output["pred"], gt))


if __name__ == '__main__':
    main()
