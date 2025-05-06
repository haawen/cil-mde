import torch
from lib.model.backbone import Backbone
from argparse import Namespace
from torchvision.transforms import ToTensor
from PIL import Image
import numpy as np

args = Namespace(
    mode='SDR',
    embed_dim=64          # Match this to configuration
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)
model = Backbone(args).to(device)
model.eval()

# 3. Load your dataset (example with a single sample)
def load_sample(rgb_path, depth_path, depth_md_path=None):
    rgb = Image.open(rgb_path).convert('RGB')
    rgb = ToTensor()(rgb).unsqueeze(0)  # Shape: (1, 3, H, W)

    """
    Commented out for now, comment in when we have sparse depth map

    # Load sparse depth map
    depth = np.load(depth_path)  # Assuming depth is stored as a .npy file
    depth = torch.tensor(depth).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)
    """
    ##################################
    # Load sparse depth map (for testing purposes, we create a random sparse depth map)
    # Comment this out when using real sparse depth map
    ##################################
    init_depth = np.load(depth_md_path)  # Load initial depth estimation
    mask = np.random.rand(*init_depth.shape) < 0.1  # Keep 10% of the depth values
    sparse_depth = np.zeros_like(init_depth)
    sparse_depth[mask] = init_depth[mask]  # Retain only the sampled values

    depth = torch.tensor(sparse_depth).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)


    # Load initial depth estimation (if in SDR mode)
    depth_md = None
    if depth_md_path:
        depth_md = np.load(depth_md_path)
        depth_md = torch.tensor(depth_md).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)

    return rgb, depth, depth_md

rgb_path = r"test\test_000000_rgb.png"  # RGB image
depth_path = r"da2_pipe_preds_npy\test_000000_depth.npy" # sparse depth map, for now not used
depth_md_path = r"da2_pipe_preds_npy\test_000000_depth.npy"  # init depth estimation

rgb, depth, depth_md = load_sample(rgb_path, depth_path, depth_md_path)

rgb = rgb.to(device).float()
depth = depth.to(device).float()
depth_md = depth_md.to(device).float() if depth_md is not None else None

with torch.no_grad():
    init_depth, guide = model(rgb=rgb, depth=depth, depth_MD=depth_md)

print("Initial Depth Shape:", init_depth.shape if init_depth is not None else None)
print("Guide Shape:", guide.shape)

if init_depth is not None:
    np.save("output_init_depth.npy", init_depth.cpu().numpy()) # We should not get this since we already have initial depth

#np.save("output_guide.npy", guide.cpu().numpy()) # Save the guide output, if needed



