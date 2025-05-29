"""
Run the MSPN refinement after getting the guidance features
"""
from argparse import Namespace
import os
import random
import torch
import numpy as np
from matplotlib import pyplot as plt
from torchvision.transforms import ToTensor
from PIL import Image
from MSPN_SDR.lib.model.MSPN import MSPN

args = Namespace(
    data_name='NYU',
    mode='SDR',
    embed_dim=64,
    pretrain='./MSPN_SDR/test_models/udr.pth',
    prop_time=6,
)
# Add your own paths to these directories
RGB_DIR = os.path.join('..', 'data','train','train')
DEPTH_MD_DIR = os.path.join('..', 'data', 'mean_train')
TOTAL_VARIANCE_DIR = os.path.join('..', 'data', 'total_var_train')


def load_sample(rgb_path, var_path, depth_md_path=None):
    rgb = Image.open(rgb_path).convert('RGB')
    rgb = ToTensor()(rgb)#.unsqueeze(0)  # Shape: (1, 3, H, W)


    #Commented out for now, comment in when we have sparse depth map

    # Load sparse depth map
    var = np.load(var_path)  # Assuming depth is stored as a .npy file
    var = torch.tensor(var)#.unsqueeze(0)  # Shape: (1, 1, H, W)

    # Load initial depth estimation (if in SDR mode)
    depth_md = None
    if depth_md_path:
        depth_md = np.load(depth_md_path)
        depth_md = torch.tensor(depth_md)#.unsqueeze(0)  # Shape: (1, 1, H, W)

    return rgb, var, depth_md

def sample_depth_from_var(total_var, depth_md, threshold=0.05):
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


def get_mask_from_var(total_var, threshold=0.05):
    return (total_var < threshold) * 1.0


def load_inputs(rgb_path, var_path, depth_md_path, threshold=0.05):
    rgb, variance, depth_md = load_sample(rgb_path, var_path, depth_md_path)
    #depth, mask_init = depth_and_mask_from_rand(depth_md)
    depth = sample_depth_from_var(variance, depth_md, threshold=threshold)

    return rgb, depth, depth_md, variance

def prepare_inputs(rgb, s_depth, depth_md, variance, guidance_net, device):
    s_depth = s_depth.to(device)
    depth_md = depth_md.to(device)
    variance = variance.to(device)

    with torch.no_grad():
        #with torch.cuda.amp.autocast():
            _, guide = guidance_net(rgb, s_depth, depth_md)

    return depth_md, guide, s_depth, variance


def load_model(model, arg):
    model.load_state_dict(torch.load(arg.pretrain))
    print(f'Checkpoint loaded from {arg.pretrain}!')
    return model


def visualize_output(y_inter, var_inter, sample_num):

    gt = np.load(os.path.join(RGB_DIR, f'sample_{sample_num}_depth.npy'))
    gt = torch.tensor(gt).unsqueeze(0)

    def get_from_func(ll, py_func, torch_func):
        return py_func([torch_func(l) for l in ll])

    var_min = get_from_func(var_inter, min, torch.min)
    var_max = get_from_func(var_inter, max, torch.max)

    depth_min = min(get_from_func(y_inter, min, torch.min), gt.min())
    depth_max = max(get_from_func(y_inter, max, torch.max), gt.max())

    fig = plt.figure()
    plt.tight_layout()
    print("Variances:")
    for i, yy in enumerate(var_inter):
        fig.add_subplot(5, len(var_inter)//2 + 1, i+1)
        plt.imshow(yy.squeeze(0).squeeze(0).cpu().numpy(), vmin=var_min, vmax=var_max, cmap='plasma')
        #plt.show()
    print("Depths:")
    for i, yy in enumerate(y_inter):
        fig.add_subplot(5, len(y_inter)//2 + 1, i + len(y_inter) + 2)
        plt.imshow(yy.squeeze(0).squeeze(0).cpu().numpy(), vmin=depth_min, vmax=depth_max, cmap='plasma')
        #plt.show()
    fig.add_subplot(5, len(var_inter)//2 + 1, 4 * (len(var_inter)//2 + 1) + 1)
    print("Ground Truth:")
    plt.imshow(gt.squeeze(0).squeeze(0).cpu().numpy(), vmin=depth_min, vmax=depth_max, cmap='plasma')
    plt.show()


def get_model_output(model, pred_init, var_init, guide):

    with torch.no_grad():
        y_inter, var_inter, _gain = model(
            pred_init,
            var_init,
            guide
        )

    return y_inter, var_inter

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

    load_model(model, args)

    SAMPLE_NUM = random.choice(os.listdir(TOTAL_VARIANCE_DIR))[7:13]
    #SAMPLE_NUM = '009067'

    # RGB image
    rgb_file = os.path.join(RGB_DIR, f'sample_{SAMPLE_NUM}_rgb.png')
    # total variance map
    var_file = os.path.join(TOTAL_VARIANCE_DIR, f'sample_{SAMPLE_NUM}_depth_var_total.npy')
    # init depth estimation
    depth_md_file = os.path.join(DEPTH_MD_DIR, f'sample_{SAMPLE_NUM}_depth_mean.npy')

    guidance_net = torch.load(os.path.join('.', 'MSPN_SDR', 'test_models', 'guidance_net.pt'))
    guidance_net = guidance_net.to(device).eval()

    rgb, s_depth, depth_md, variance = load_inputs(rgb_file, var_file, depth_md_file)
    rgb = rgb.to(device)

    # "Batch Size 1"
    rgb = rgb.unsqueeze(0)
    s_depth = s_depth.unsqueeze(0)
    depth_md = depth_md.unsqueeze(0)
    variance = variance.unsqueeze(0)

    pred_init, guide, s_depth, var_init = prepare_inputs(rgb, s_depth, depth_md, variance, guidance_net, device)

    with torch.no_grad():
        pred_inter, var_inter, _gain = model(
            pred_init,
            var_init,
            guide
        )

    visualize_output(pred_inter, var_inter, SAMPLE_NUM)

    gt = np.load(os.path.join(RGB_DIR, f'sample_{SAMPLE_NUM}_depth.npy'))
    gt = torch.tensor(gt).unsqueeze(0).unsqueeze(0).to(device)

    print(eval_depth(pred_init, gt))
    print(eval_depth(pred_inter[-1], gt))


if __name__ == '__main__':
    main()
