from argparse import Namespace
import os
import torch
from torchvision.transforms import ToTensor
from PIL import Image
import numpy as np
from .lib.model.backbone import Backbone

# 3. Load your dataset (example with a single sample)
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


def main():
    args = Namespace(
        mode='SDR',
        embed_dim=64          # Match this to configuration
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    model = Backbone(args).to(device)
    model.eval()

    rgb_path = os.path.join("test_images", "test_000000_rgb.png")  # RGB image
    depth_path = os.path.join("da2_pipe_preds_npy", "test_000000_depth.npy") # sparse depth map, for now not used
    depth_md_path = os.path.join("da2_pipe_preds_npy", "test_000000_depth.npy")  # init depth estimation

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

if __name__ == '__main__':
    main()