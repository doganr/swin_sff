"""
Swin-SFF shape from focus strategies.

Author: doganr
Contact: dogan@trabzon.edu.tr
Last updated: 2024-06-28
"""

import torch
import torch.nn.functional as F

def spatial_frequency(fs, device, kernel_radius=5):
    """
    Perform spatial frequency analysis on a tensor of feature maps.
    
    Parameters:
        fs (Tensor): Input tensor of shape [n, 96, 224, 224].
        device (torch.device): The device (CPU or GPU) to perform computations on.
        kernel_radius (int): Radius of the kernel for spatial frequency calculation.
        
    Returns:
        np.ndarray: Decision map of shape [n, 224, 224] with integer values.
    """
    # Ensure the tensor is on the correct device
    fs = fs.to(device)
    
    n, c, h, w = fs.shape

    # Define shift kernels
    r_shift_kernel = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 0, 0]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    b_shift_kernel = torch.tensor([[0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    
    # Repeat the kernels for each channel
    r_shift_kernel = r_shift_kernel.repeat(c, 1, 1, 1)
    b_shift_kernel = b_shift_kernel.repeat(c, 1, 1, 1)

    # Apply convolution with right shift kernel
    fs_r_shift = F.conv2d(fs, r_shift_kernel, padding=1, groups=c)
    # Apply convolution with bottom shift kernel
    fs_b_shift = F.conv2d(fs, b_shift_kernel, padding=1, groups=c)
    
    # Calculate the gradients
    fs_grad = torch.pow((fs_r_shift - fs), 2) + torch.pow((fs_b_shift - fs), 2)
    
    # Define kernel for summing spatial frequencies
    kernel_size = kernel_radius * 2 + 1
    add_kernel = torch.ones((c, 1, kernel_size, kernel_size), dtype=torch.float32, device=device)
    kernel_padding = kernel_size // 2
    
    # Sum up gradients
    fs_sf = torch.sum(F.conv2d(fs_grad, add_kernel, padding=kernel_padding, groups=c), dim=1)
    
    # Stack and find max indices
    dm_tensor = torch.argmax(fs_sf, dim=0)
    decision_map = dm_tensor.squeeze().cpu().numpy().astype(int)
    
    return decision_map

def tenengrad(fs, device):
    """
    Perform channel selection based on Tenengrad method for a tensor of feature maps.
    
    Parameters:
        fs (Tensor): Input tensor of shape [n, 96, 224, 224].
        device (torch.device): The device (CPU or GPU) to perform computations on.
        
    Returns:
        np.ndarray: Decision map of shape [n, 224, 224] with integer values.
    """
    # Ensure the tensor is on the correct device
    fs = fs.to(device)
    
    n, c, h, w = fs.shape

    # Define Sobel filters
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    
    sobel_x = sobel_x.repeat(c, 1, 1, 1)
    sobel_y = sobel_y.repeat(c, 1, 1, 1)

    # Apply Sobel filters to compute gradients
    grad_x = F.conv2d(fs, sobel_x, padding=1, groups=c)
    grad_y = F.conv2d(fs, sobel_y, padding=1, groups=c)

    # Compute gradient magnitude
    grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)  # Shape [n, 96, 224, 224]

    # Compute the Tenengrad score by averaging the gradient magnitudes across channels
    tenengrad_scores = grad_magnitude.mean(dim=1)  # Shape [n, 224, 224]

    # Find the image with the maximum Tenengrad score for each pixel
    decision_map = torch.argmax(tenengrad_scores, dim=0)  # Decision map shape [224, 224]

    # Convert decision map to CPU and to numpy
    decision_map_np = decision_map.cpu().numpy().astype(int)
    
    return decision_map_np

def energy_of_laplacian(fs, device):
    """
    Perform image selection based on the Energy of Laplacian method for a tensor of feature maps.
    
    Parameters:
        fs (Tensor): Input tensor of shape [n, 96, 224, 224].
        device (torch.device): The device (CPU or GPU) to perform computations on.
        
    Returns:
        np.ndarray: Decision map of shape [n, 224, 224] with integer values.
    """
    # Ensure the tensor is on the correct device
    fs = fs.to(device)
    
    n, c, h, w = fs.shape

    # Define Laplacian filter
    laplacian_filter = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    laplacian_filter = laplacian_filter.repeat(c, 1, 1, 1)

    # Apply Laplacian filter to compute Laplacian of the image
    laplacian = F.conv2d(fs, laplacian_filter, padding=1, groups=c)  # Shape [n, 96, 224, 224]

    # Compute the energy of Laplacian by summing the absolute values
    energy = torch.sum(torch.abs(laplacian), dim=1)  # Shape [n, 224, 224]

    # Find the image with the maximum energy of Laplacian for each pixel
    decision_map = torch.argmax(energy, dim=0)  # Decision map shape [224, 224]

    # Convert decision map to CPU and to numpy
    decision_map_np = decision_map.cpu().numpy().astype(int)
    
    return decision_map_np

def variance(fs, device):
    """
    Perform channel selection based on variance for a tensor of feature maps.
    
    Parameters:
        fs (Tensor): Input tensor of shape [n, 96, 224, 224].
        device (torch.device): The device (CPU or GPU) to perform computations on.
        
    Returns:
        np.ndarray: Decision map of shape [n, 224, 224] with integer values.
    """
     # Ensure the tensor is on the correct device
    fs = fs.to(device)
    
    n, c, h, w = fs.shape

    # Compute the mean and variance for each pixel across channels
    mean_fs = fs.mean(dim=1, keepdim=True)  # Mean per pixel location, shape [n, 1, 224, 224]
    
    # Compute the variance per channel at each pixel
    variances = torch.var(fs - mean_fs, dim=1, keepdim=False)  # Variance along channels, shape [n, c, 224, 224]

    # Find the channel with the maximum variance for each pixel
    decision_map = torch.argmax(variances, dim=0)  # Decision map shape [n, 224, 224]

    # Convert decision map to CPU and to numpy
    decision_map_np = decision_map.cpu().numpy().astype(int)
    
    return decision_map_np