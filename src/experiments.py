import argparse
import os
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import ToTensor

from Depth_Anything_V2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2
from MSPN_SDR.lib.model.MSPN import MSPN
from evaluate import run_evaluation
from run_refinement import compute_aurg, compute_ause, sample_depth_from_var


EMBED_DIM = 64
PROP_TIME = 6

# Prepare the transform pipeline (same as for test)
transform_pipeline = transforms.Compose([
    transforms.Resize((518, 518),
                      interpolation=transforms.InterpolationMode.BILINEAR,
                      antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Original image size (height, width)
original_size_hw = (426, 560)

def init_udr(device):
    model = MSPN(EMBED_DIM, PROP_TIME)
    model = model.to(device)
    model.load_state_dict(torch.load('./checkpoints/udr.pth'))
    return model

def init_guidance_model(device):
    guidance_net = torch.load(os.path.join('./checkpoints/guidance_net.pt'))
    guidance_net = guidance_net.to(device)
    guidance_net = guidance_net.eval()
    return guidance_net

def init_var_model(device):
    model_configs = {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
    max_depth = 10
    var_model_path = "./checkpoints/variance-model2025-05-17-18-54-06"

    var_model = DepthAnythingV2(**{**model_configs, 'max_depth': max_depth}).to(device)
    var_model.load_state_dict(torch.load(var_model_path))
    var_model.eval()
    return var_model

def init_depth_models(device):
    model_configs = {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
    max_depth = 10

    saved_model_paths = ["./checkpoints/model2025-05-17-13-07-05",
                         "./checkpoints/model2025-05-17-13-07-26",
                         "./checkpoints/model2025-05-17-16-09-25",
                         "./checkpoints/model2025-05-17-16-09-48"
                        ]
    models = []
    for path in saved_model_paths:
        model = DepthAnythingV2(**{**model_configs, 'max_depth': max_depth}).to(device)
        model.load_state_dict(torch.load(path))
        model.eval()
        models.append(model)

    return models


def depth_var_inference(depth_models, var_model, image, device, split_var=False, w_flipping=False):
    img = transform_pipeline(image)
    with torch.inference_mode():
        img = img.unsqueeze(0).to(device)
        preds = [model(img) for model in depth_models]
        pred_stack = torch.stack(preds, dim=0)

        pred_mean = pred_stack.mean(dim=0)  # Shape: (1, H, W)
        pred_epi_var = pred_stack.var(dim=0)    # Shape: (1, H, W)
        pred_al_var = var_model(img)
        pred_total_var = pred_epi_var + pred_al_var
        pred_var_w_flipping = torch.square(torch.flip(depth_models[0](torch.flip(img, dims=[-1])), dims=[-1]) - depth_models[0](img))

        # Resize to original size
        pred_mean = F.interpolate(pred_mean.unsqueeze(0),
                                  size=original_size_hw,
                                  mode='bilinear',
                                  align_corners=False)
        pred_epi_var = F.interpolate(pred_epi_var.unsqueeze(0),
                                     size=original_size_hw,
                                     mode='bilinear',
                                     align_corners=False)
        pred_al_var = F.interpolate(pred_al_var.unsqueeze(0),
                                    size=original_size_hw,
                                    mode='bilinear',
                                    align_corners=False)
        pred_total_var = F.interpolate(pred_total_var.unsqueeze(0),
                                       size=original_size_hw,
                                       mode='bilinear',
                                       align_corners=False)
        pred_var_w_flipping = F.interpolate(pred_var_w_flipping.unsqueeze(0),
                                            size=original_size_hw,
                                            mode='bilinear',
                                            align_corners=False)


    output_list = [pred_mean]

    if split_var:
        output_list.append(pred_epi_var)
        output_list.append(pred_al_var)
    else:
        output_list.append(pred_total_var)

    if w_flipping:
        output_list.append(pred_var_w_flipping)

    return tuple(output_list)


# Compute AUSE and AURG of output variance
def experiment_1(data_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    depth_models = init_depth_models(device)
    var_model = init_var_model(device)

    # Use last 10% of training set as validation set
    with open(os.path.join(data_dir, 'train_list.txt'), 'r') as f:
        all_samples = [s.strip() for s in sorted(list(f))]
    mid = int(len(all_samples)*0.9)
    tail_samples = all_samples[mid:]

    train_data_dir = os.path.join(data_dir, 'train', 'train')

    AUSE = {'total': {'l1': [], 'l2': []}, 'flipping': {'l1': [], 'l2': []}}
    AURG = {'total': {'l1': [], 'l2': []}, 'flipping': {'l1': [], 'l2': []}}

    for filenames in tqdm(tail_samples):
        sample_num = filenames.strip()[7:13]
        img_file = os.path.join(train_data_dir, f'sample_{sample_num}_rgb.png')
        img = Image.open(img_file).convert('RGB')

        gt = np.load(os.path.join(train_data_dir, f'sample_{sample_num}_depth.npy'))
        gt = gt[np.newaxis, ...]


        pred_mean, pred_total_var, pred_var_w_flipping = depth_var_inference(depth_models, var_model, img, device, w_flipping=True)
        l1_error = np.abs(pred_mean.view(-1).cpu().numpy() - gt.flatten())
        l2_error = l1_error ** 2
        var = {'total': pred_total_var.view(-1).cpu().numpy(), 'flipping': pred_var_w_flipping.view(-1).cpu().numpy()}
        error = {'l1': l1_error, 'l2': l2_error}

        for var_type in ['total', 'flipping']:
            for error_type in ['l1', 'l2']:
                AUSE[var_type][error_type].append(compute_ause(error[error_type], var[var_type]))
                AURG[var_type][error_type].append(compute_aurg(error[error_type], var[var_type]))

    for var_type in ['total', 'flipping']:
        for error_type in ['l1', 'l2']:
            print(f"Mean AUSE {error_type.capitalize()} ({var_type}): {np.mean(AUSE[var_type][error_type]):.4f}")
            print(f"Mean AURG {error_type.capitalize()} ({var_type}): {np.mean(AURG[var_type][error_type]):.4f}")




# Run evaluation on chosen model
def experiment_2(data_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    udr_model = init_udr(device)
    guidance_net = init_guidance_model(device)
    depth_models = init_depth_models(device)
    var_model = init_var_model(device)

    # Use last 10% of training set as validation set
    with open(os.path.join(data_dir, 'train_list.txt'), 'r') as f:
        all_samples = [s.strip() for s in sorted(list(f))]
    mid = int(len(all_samples)*0.9)
    tail_samples = all_samples[mid:]

    train_data_dir = os.path.join(data_dir, 'train', 'train')

    initial_output_triples = []
    output_triples = []
    
    for filenames in tqdm(tail_samples):
        sample_num = filenames.strip()[7:13]
        img_file = os.path.join(train_data_dir, f'sample_{sample_num}_rgb.png')
        img = Image.open(img_file).convert('RGB')

        gt = np.load(os.path.join(train_data_dir, f'sample_{sample_num}_depth.npy'))
        gt = gt[np.newaxis, np.newaxis, ...]


        pred_mean, pred_total_var = depth_var_inference(depth_models, var_model, img, device)
        sampled_depth = sample_depth_from_var(pred_total_var, pred_mean, threshold=0.05)

        with torch.inference_mode():
            img = ToTensor()(img)
            img = img.to(device).unsqueeze(0)
            sampled_depth = sampled_depth.to(device)

            _, guide = guidance_net(img, sampled_depth, pred_mean)

            preds, _, _ = udr_model(pred_mean, pred_total_var, guide)
        initial_output_triples.append((preds[0].cpu().numpy(), gt, sample_num))
        output_triples.append((preds[-1].cpu().numpy(), gt, sample_num))
    run_evaluation(initial_output_triples, "EnsembleDA2")
    run_evaluation(output_triples, "UDR")

# Read Kalman Gain
def experiment_3(data_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    udr_model = init_udr(device)
    guidance_net = init_guidance_model(device)
    depth_models = init_depth_models(device)
    var_model = init_var_model(device)

    # Use last 10% of training set as validation set
    with open(os.path.join(data_dir, 'train_list.txt'), 'r') as f:
        all_samples = [s.strip() for s in sorted(list(f))]
    mid = int(len(all_samples)*0.9)
    tail_samples = all_samples[mid:]

    train_data_dir = os.path.join(data_dir, 'train', 'train')


    sample_nums = []
    output_gains = []
    for i in range(PROP_TIME):
        output_gains.append([])

    for filenames in tqdm(tail_samples):
        sample_num = filenames.strip()[7:13]
        img_file = os.path.join(train_data_dir, f'sample_{sample_num}_rgb.png')
        img = Image.open(img_file).convert('RGB')

        pred_mean, pred_total_var = depth_var_inference(depth_models, var_model, img, device)
        sampled_depth = sample_depth_from_var(pred_total_var, pred_mean, threshold=0.05)

        with torch.inference_mode():
            img = ToTensor()(img)
            img = img.to(device).unsqueeze(0)
            sampled_depth = sampled_depth.to(device)

            _, guide = guidance_net(img, sampled_depth, pred_mean)

            _, _, gains = udr_model(pred_mean, pred_total_var, guide)
        sample_nums.append(sample_num)
        for i, gain in enumerate(gains):
            output_gains[i].append(gain.cpu().numpy())


    print('=== Min/Mean/Max Kalman gain per Iteration ===')
    print(18*' ' + 'min' + 5*' ' + 'mean' + 6*' ' + 'max')
    for i, it in enumerate(output_gains):
        print(f"Iteration {i+1}:   {np.min(it):.4f}   {np.mean(it):.4f}   {np.max(it):.4f}")

    num_images = 6
    _, axes = plt.subplots(num_images, PROP_TIME, figsize=(2 * num_images, 10))
    for i, it in enumerate(output_gains):
        head = it[:num_images]
        for j, gain in enumerate(head):
            axes[j, i].imshow(np.minimum(-np.log(gain[0][0]), 1), cmap='plasma', vmin=0.0, vmax=1.0)
            axes[j, i].set_title(f'Sample {sample_nums[j]}, It {i+1}')
            axes[j, i].axis('off')
    plt.tight_layout()
    plt.show()


# Generate Predictions
def experiment_4(data_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    udr_model = init_udr(device)
    guidance_net = init_guidance_model(device)
    depth_models = init_depth_models(device)
    var_model = init_var_model(device)

    # Use last 10% of training set as validation set
    with open(os.path.join(data_dir, 'test_list.txt'), 'r') as f:
        test_samples = [s.strip() for s in f]#sorted(list(f))]
    #mid = int(len(all_samples)*0.9)
    #tail_samples = all_samples[mid:]

    test_data_dir = os.path.join(data_dir, 'test', 'test')
    predictions_dir = os.path.join(data_dir, 'output', 'predictions')
    for filenames in tqdm(test_samples):
        sample_num = filenames.strip()[5:11]
        img_file = os.path.join(test_data_dir, f'test_{sample_num}_rgb.png')
        img = Image.open(img_file).convert('RGB')

        pred_mean, pred_total_var = depth_var_inference(depth_models, var_model, img, device)
        sampled_depth = sample_depth_from_var(pred_total_var, pred_mean, threshold=0.05)

        with torch.inference_mode():
            img = ToTensor()(img)
            img = img.to(device).unsqueeze(0)
            sampled_depth = sampled_depth.to(device)

            _, guide = guidance_net(img, sampled_depth, pred_mean)

            preds, _, _ = udr_model(pred_mean, pred_total_var, guide)

            output = F.interpolate(
                preds[-1],
                size=(426, 560),  # Original input dimensions
                mode='bilinear',
                align_corners=True
            )
            filename = f'test_{sample_num}_depth.npy'

            # Save depth map prediction as numpy array
            depth_pred = output.cpu().squeeze().numpy()
            np.save(os.path.join(predictions_dir, filename), depth_pred)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default='../data')
    parser.add_argument("--experiment_number", type=int, default=1)

    args = parser.parse_args()

    if not args.data_dir:
        raise NotADirectoryError("Please add a valid data directory.")
    
    data_dir = os.path.normpath(args.data_dir)

    if args.experiment_number == 1:
        experiment_1(data_dir)
    elif args.experiment_number == 2:
        experiment_2(data_dir)
    elif args.experiment_number == 3:
        experiment_3(data_dir)
    elif args.experiment_number == 4:
        experiment_4(data_dir)
    else:
        raise ValueError("Please run an experiment from the set {1, 2, 3}.")



if __name__ == '__main__':
    main()
