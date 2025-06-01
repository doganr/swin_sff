"""
Swin-SFF model testing.

Author: doganr
Contact: dogan@trabzon.edu.tr
Last updated: 2024-06-28
"""

import os
import argparse
import torch
import numpy as np
from PIL import Image
import plotly.graph_objects as go

from swin_sff import SwinSFF

def load_model(path, in_chans, out_chans, device):
    """
    Loads a trained SwinSFF model from the specified path.

    Args:
        path (str): Path to the saved model (.pth) file.
        in_chans (int): Number of input channels expected by the model.
        out_chans (int): Number of output channels produced by the model.
        device (torch.device): Device to which the model should be moved (e.g., 'cuda' or 'cpu').

    Returns:
        SwinSFF: The loaded and initialized SwinSFF model set to evaluation mode.

    Notes:
        - Model weights are loaded with strict=False to allow partial compatibility.
        - Prints model name and source path for verification.
    """
    SwinSFF_model = SwinSFF(in_chans=in_chans, out_chans=out_chans)
    SwinSFF_model.load_state_dict(torch.load(path), False)

    para = sum([np.prod(list(p.size())) for p in SwinSFF_model.parameters()])
    type_size = 4
    print(f'Model {SwinSFF_model._get_name()} loaded from [{path}].')

    SwinSFF_model.eval()
    SwinSFF_model.to(device)

    return SwinSFF_model

def select_pixels(x,y):
    """
    Selects specific pixels from a multi-channel 3D array based on corresponding index values.

    Args:
        x (np.ndarray): Input array of shape (W, H, ...) representing a multi-channel image or feature map.
        y (np.ndarray): Index array of shape (W, H) indicating which channel to select for each (i, j) location.

    Returns:
        np.ndarray: An array of shape (W, H, ...) where the channel at each position is selected based on `y`.

    Notes:
        - Assumes that `x` has at least 3 dimensions and `y` contains valid indices along the last dimension of `x`.
        - Useful for extracting per-pixel best-focus slices or attention-selected channels.
    """
    w = x.shape[0]
    h = x.shape[1]
    x_tmp = x.reshape(w*h,*x.shape[2:])
    y_tmp = y.ravel()
    x_dest = np.array([xi[yi] for xi,yi in zip(x_tmp,y_tmp)])
    nshape = (w,h) if x_dest.ndim == 1 else (w,h,*x_dest.shape[1:])
    x_dest = x_dest.reshape(*nshape)
    return x_dest

def mean_filter(z, kernel_size=3):
    """
    Applies a mean (average) filter to a 2D array for smoothing.

    Args:
        z (np.ndarray): Input 2D array (e.g., depth map or grayscale image).
        kernel_size (int): Size of the square kernel. Must be an odd number (default: 3).

    Returns:
        np.ndarray: Smoothed 2D array of the same shape as the input.

    Notes:
        - Uses 'reflect' padding to handle border effects.
        - Larger kernel sizes produce stronger smoothing but may blur details.
    """
    k = kernel_size // 2
    z_padded = np.pad(z, k, mode='reflect')
    z_filtered = np.zeros_like(z)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            z_filtered[i, j] = np.mean(z_padded[i:i + kernel_size, j:j + kernel_size])
    return z_filtered

def show_3d_plot(z, fused, output_html='3d_plot.html', z_min_value=None, z_max_value=None):
    """
    Generates and saves a 3D surface plot of the depth map with color texture.

    Args:
        z (np.ndarray): 2D array representing depth values.
        fused (np.ndarray): 3D RGB image (H x W x 3) to be used for surface coloring.
        output_html (str): Output file path for the HTML plot (default: '3d_plot.html').
        z_min_value (float, optional): Minimum value for the Z-axis range. If None, computed from `z`.
        z_max_value (float, optional): Maximum value for the Z-axis range. If None, computed from `z`.

    Returns:
        None

    Notes:
        - Uses Plotly's Mesh3d to render the 3D surface.
        - RGB colors from `fused` are mapped to surface vertices.
        - The resulting interactive plot is saved as an HTML file.
    """
    x = np.arange(fused.shape[1])
    y = np.arange(fused.shape[0])
    x, y = np.meshgrid(x, y)
    z_scaled = z
    r = fused[:, :, 0] / 255.0
    g = fused[:, :, 1] / 255.0
    b = fused[:, :, 2] / 255.0
    colors = np.stack((r, g, b), axis=-1).reshape(-1, 3)
    z_min = z_min_value if z_min_value is not None else z_scaled.min()
    z_max = z_max_value if z_max_value is not None else z_scaled.max()
    vertices = np.array([x.ravel(), y.ravel(), z_scaled.ravel()]).T
    I, J, K = [], [], []
    for i in range(z.shape[0] - 1):
        for j in range(z.shape[1] - 1):
            I.append(i * z.shape[1] + j)
            J.append(i * z.shape[1] + j + 1)
            K.append((i + 1) * z.shape[1] + j)
            I.append(i * z.shape[1] + j + 1)
            J.append((i + 1) * z.shape[1] + j + 1)
            K.append((i + 1) * z.shape[1] + j)
    fig = go.Figure(data=[go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=I, j=J, k=K,
        vertexcolor=colors,
        opacity=0.8
    )])
    fig.update_layout(scene=dict(
        xaxis_title='X axis',
        yaxis_title='Y axis',
        zaxis_title='Depth',
        zaxis=dict(range=[z_min, z_max])
    ))
    fig.write_html(output_html)

def show_3d_plot_filtered(z, fused, output_html='3d_plot.html', z_min_value=None, z_max_value=None, kernel_size=3):
    """
    Generates and saves a 3D surface plot of a filtered depth map using mean filtering.

    Args:
        z (np.ndarray): 2D array representing raw depth values.
        fused (np.ndarray): 3D RGB image (H x W x 3) used for coloring the surface.
        output_html (str): Output path to save the interactive HTML plot (default: '3d_plot.html').
        z_min_value (float, optional): Minimum Z-axis value. Automatically computed if None.
        z_max_value (float, optional): Maximum Z-axis value. Automatically computed if None.
        kernel_size (int): Size of the mean filter kernel applied to `z` (default: 3).

    Returns:
        None

    Notes:
        - This function smooths the input depth map before plotting.
        - The RGB values from `fused` are mapped onto the surface mesh.
        - Saves an interactive 3D plot in HTML format using Plotly.
    """

    x = np.arange(fused.shape[1])
    y = np.arange(fused.shape[0])
    x, y = np.meshgrid(x, y)
    z_scaled = mean_filter(z, kernel_size=kernel_size)
    r = fused[:, :, 0] / 255.0
    g = fused[:, :, 1] / 255.0
    b = fused[:, :, 2] / 255.0
    colors = np.stack((r, g, b), axis=-1).reshape(-1, 3)
    z_min = z_min_value if z_min_value is not None else z_scaled.min()
    z_max = z_max_value if z_max_value is not None else z_scaled.max()
    vertices = np.array([x.ravel(), y.ravel(), z_scaled.ravel()]).T
    I, J, K = [], [], []
    for i in range(z.shape[0] - 1):
        for j in range(z.shape[1] - 1):
            I.append(i * z.shape[1] + j)
            J.append(i * z.shape[1] + j + 1)
            K.append((i + 1) * z.shape[1] + j)
            I.append(i * z.shape[1] + j + 1)
            J.append((i + 1) * z.shape[1] + j + 1)
            K.append((i + 1) * z.shape[1] + j)
    fig = go.Figure(data=[go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=I, j=J, k=K,
        vertexcolor=colors,
        opacity=0.8
    )])
    fig.update_layout(scene=dict(
        xaxis_title='X axis',
        yaxis_title='Y axis',
        zaxis_title='Depth',
        zaxis=dict(range=[z_min, z_max])
    ))
    fig.write_html(output_html)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Swin-SFF model testing script.")

    parser.add_argument('--model_path', type=str,
        default='./models/best.model',
        help="Path to the trained model file. Default is './models/best.model'."
             " Make sure this file exists and corresponds to your best checkpoint.")
    parser.add_argument('--imgs_folder', type=str, default='outputs/simu15',
        help='Folder containing test images.')
    parser.add_argument('--in_chans', type=int, default=1,
        help='Number of input channels for the model.')
    parser.add_argument('--out_chans', type=int, default=1,
        help='Number of output channels for the model.')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu',
        help='Device to run the model on.')
    parser.add_argument('--z_min', type=float, default=0, help='Minimum Z value for 3D plot.')
    parser.add_argument('--z_max', type=float, default=100, help='Maximum Z value for 3D plot.')
    parser.add_argument('--kernel_size', type=int, default=3, help='Kernel size for filtering depth map.')

    args = parser.parse_args()

    model = load_model(args.model_path, args.in_chans, args.out_chans, args.device)

    images = [os.path.join(args.imgs_folder, f) for f in os.listdir(args.imgs_folder) if os.path.isfile(os.path.join(args.imgs_folder, f))]

    z, fused = model.fuse(images)

    output_dir = os.path.join(args.imgs_folder, 'output')
    os.makedirs(output_dir, exist_ok=True)

    Image.fromarray(fused).save(os.path.join(output_dir, 'fused.png'))
    print(f"Fused image saved to '{output_dir}/fused.png'.")

    np.savetxt(os.path.join(output_dir, 'fused.txt'), z, fmt='%i', delimiter='\t')
    print(f"3D depth info saved to '{output_dir}/fused.txt'.")

    show_3d_plot(z, fused, output_html=os.path.join(output_dir, 'fused3D.html'),
                 z_min_value=args.z_min, z_max_value=args.z_max)
    print(f"3D plot saved to '{output_dir}/fused3D.html'. You can open it in your web browser.")

    show_3d_plot_filtered(z, fused, output_html=os.path.join(output_dir, 'fused3D_filtered.html'),
                          z_min_value=args.z_min, z_max_value=args.z_max, kernel_size=args.kernel_size)
    print(f"3D filtered plot saved to '{output_dir}/fused3D_filtered.html'. You can open it in your web browser.")
