"""
Swin-SFF model training.

Author: doganr
Contact: dogan@trabzon.edu.tr
Last updated: 2024-06-28
"""

import os
import argparse
import posixpath
import time
import numpy as np
from tqdm import tqdm, trange
import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from ssim import MS_SSIM
from swin_sff import SwinSFF

class CocoDataset(Dataset):
    """
    A custom PyTorch Dataset for loading grayscale images from a list of file paths.

    Args:
        image_paths (list): List of file paths to the images.
        transform (callable, optional): Optional transform to be applied on each image.

    Returns:
        torch.Tensor: Transformed image tensor.

    Notes:
        - Images are loaded in grayscale mode ('L').
    """
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('L')  
        if self.transform:
            image = self.transform(image)
        return image

def train(args, original_img_paths):
    """
    Trains the Swin-SFF model on a grayscale image dataset using a combination 
    of pixel-wise L1 loss and weighted SSIM loss.

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing all training configurations,
            including epochs, batch size, learning rate, model settings, and SSIM weight parameters.
        original_img_paths (list of str): List of file paths to grayscale input images used for training.

    Description:
        - Loads grayscale images via a custom CocoDataset class.
        - Applies resizing and tensor conversion transformations.
        - Initializes the SwinSFF model, optimizer, and loss functions.
        - Trains the model using L1 + weighted SSIM loss for a specified number of epochs.
        - Logs loss metrics periodically based on log interval.
        - Saves the model and loss history into a .pth file after training.

    Saves:
        A .pth file containing:
            - Trained model state_dict
            - Pixel loss, SSIM loss, and total loss histories
            - Selected SSIM weight
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_workers = min(args.num_workers, os.cpu_count() or 1)
    ssim_path = [f"1e{int(np.log10(w))}" for w in args.ssim_weights]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = CocoDataset(original_img_paths, transform=transform)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    model = SwinSFF(in_chans=args.in_chans, out_chans=args.out_chans)

    optimizer = Adam(model.parameters(), args.lr)
    l1_loss = torch.nn.L1Loss()
    ssim_loss = MS_SSIM(data_range=1.0, size_average=True, channel=args.in_chans).to(device)

    model.to(device)
    
    tbar = trange(args.epochs, leave=True, dynamic_ncols=True, 
                  bar_format="Epoch {n_fmt}/{total_fmt} |{bar:40}| {percentage:3.0f}% {postfix}")

    tqdm.write(f'BATCH SIZE {args.batch_size}.')
    tqdm.write(f'Train images number {len(original_img_paths)}.')
    tqdm.write(f'Train images samples {int(len(original_img_paths)/args.batch_size)}\n')
    tqdm.write('Start training.....')    

    Loss_pixel = []
    Loss_ssim = []
    Loss_all = []
    all_ssim_loss = 0.
    all_pixel_loss = 0.

    for epoch in tbar:
        model.train()
        for batch_idx, img in enumerate(data_loader):
            optimizer.zero_grad()
            img = img.to(device)

            outputs = model.autoencoder(img)

            x = img.clone().detach()

            pixel_loss_value = l1_loss(outputs, x)
            ssim_loss_value = 1 - ssim_loss(outputs, x)

            total_loss = pixel_loss_value + args.ssim_weights[args.ssim_index] * ssim_loss_value
            total_loss.backward()
            optimizer.step()

            all_ssim_loss += ssim_loss_value.item()
            all_pixel_loss += pixel_loss_value.item()

            if (batch_idx + 1) % args.log_interval == 0:
                mesg = (f"Batch {batch_idx}/{len(data_loader)}, "
                        f"Pixel Loss: {all_pixel_loss / args.log_interval:.6f}, "
                        f"SSIM Loss: {all_ssim_loss / args.log_interval:.6f}, "
                        f"Total Loss: {(args.ssim_weights[args.ssim_index] * all_ssim_loss + all_pixel_loss) / args.log_interval:.6f}")
                tbar.set_postfix_str(mesg)
                Loss_pixel.append(all_pixel_loss / args.log_interval)
                Loss_ssim.append(all_ssim_loss / args.log_interval)
                Loss_all.append((args.ssim_weights[args.ssim_index] * all_ssim_loss + all_pixel_loss) / args.log_interval)
                all_ssim_loss = 0.
                all_pixel_loss = 0.

    model.eval()
    model.cpu()

    save_data = {
        'epoch': args.epochs,
        'ssim_weight': args.ssim_weights[args.ssim_index],
        'model_state_dict': model.state_dict(),
        'loss_pixel': Loss_pixel,
        'loss_ssim': Loss_ssim,
        'loss_total': Loss_all
    }

    timestamp = time.ctime().replace(' ', '_').replace(':', '_')
    save_file_name = f"Final_model_and_losses_epoch_{args.epochs}_{timestamp}_{ssim_path[args.ssim_index]}.pth"
    save_file_path = posixpath.join(args.save_model_dir, save_file_name)

    torch.save(save_data, save_file_path)

    tqdm.write(f"\nDone, trained model and losses saved at {save_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for model with COCO dataset.")

    parser.add_argument('--original_imgs_path', type=str, default='./coco/train2017/',
                        help='Path to the directory containing the original training images (default: ./coco/train2017/ )')
    parser.add_argument('--save_model_dir', type=str, default='./models',
                        help='Directory to save the trained model.')       
    parser.add_argument('--epochs', type=int, default=25, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--in_chans', type=int, default=1, help='Number of input channels for the model.')
    parser.add_argument('--out_chans', type=int, default=1, help='Number of output channels for the model.')
    parser.add_argument('--log_interval', type=int, default=5, help='How often to log training metrics (in steps).')   
    parser.add_argument('--num_workers', type=int, default=4, 
                        help='Number of worker processes for data loading. '
                        'Will be clamped to available CPU cores.') 
    parser.add_argument('--ssim_weights', nargs='+', type=float, default=[1, 10, 100, 1000, 10000],
                        help='List of SSIM weights to choose from by index. Must be powers of 10 '
                            '(e.g., 1, 10, 100...) to ensure correct path tag generation.')
    parser.add_argument('--ssim_index', type=int, default=2,
                        help='Index of SSIM weight to use (from --ssim_weights). Must be in [0, len(ssim_weights)-1].')

    args = parser.parse_args()

    if args.ssim_index < 0 or args.ssim_index >= len(args.ssim_weights):
        raise ValueError(f"--ssim_index must be between 0 and {len(args.ssim_weights) - 1}")

    coco_images = [posixpath.join(args.original_imgs_path, img) for img in os.listdir(args.original_imgs_path) if img.endswith('.jpg') or img.endswith('.png')]

    train(args, coco_images)
