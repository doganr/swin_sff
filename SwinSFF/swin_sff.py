"""
Swin-SFF model for shape from focus.

Author: doganr
Contact: dogan@trabzon.edu.tr
Last updated: 2024-06-28
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import trunc_normal_
import numpy as np
from PIL import Image
from tqdm import tqdm

from swin_transformer import SwinTransformerBlock, PatchEmbed
from sff_strategies import spatial_frequency

class StageModule(nn.Module):
    """
    A Swin Transformer stage composed of several SwinTransformerBlocks.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple): Resolution of the input feature map.
        depth (int): Number of Swin Transformer blocks in the stage.
        num_heads (int): Number of attention heads.
        window_size (int): Size of the attention window.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        qkv_bias (bool): If True, add a learnable bias to query, key, value.
        qk_scale (float or None): Override default QK scale of sqrt(dim).
        drop (float): Dropout rate.
        attn_drop (float): Attention dropout rate.
        drop_path (float or list): Stochastic depth rate.
        norm_layer (nn.Module): Normalization layer.
        use_checkpoint (bool): If True, use checkpointing to save memory.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
            return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

class SwinSFF(nn.Module):
    """
    SwinSFF: A transformer-based autoencoder model for shape-from-focus imaging.

    This model extracts focus-related depth information and reconstructs fused, focused images
    by using Swin Transformer stages and shallow decoding.

    Args:
        img_size (int): Size of input image.
        patch_size (int): Size of image patches.
        in_chans (int): Number of input channels.
        out_chans (int): Number of output channels.
        embed_dim (int): Embedding dimension.
        depths (list): Number of blocks at each Swin Transformer stage.
        num_heads (list): Number of attention heads per stage.
        window_size (int): Window size for local attention.
        mlp_ratio (float): MLP hidden dim to embedding dim ratio.
        qkv_bias (bool): Whether to use learnable bias for QKV.
        qk_scale (float): Custom QK scale factor.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Dropout rate for attention weights.
        drop_path_rate (float): Stochastic depth rate.
        norm_layer (nn.Module): Normalization layer.
        ape (bool): Whether to use absolute position embedding.
        patch_norm (bool): Whether to apply normalization after patch embedding.
        use_checkpoint (bool): Whether to use checkpointing for memory efficiency.
    """
    def __init__(self, img_size=224, patch_size=1, in_chans=1, out_chans=1,
                 embed_dim=96, depths=[6, 6, 6], num_heads=[1, 2, 4],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()

        self.out_chans = out_chans
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim)
        self.num_features_up = int(embed_dim)
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        #  encoder
        self.EN1_0 = StageModule(dim=int(embed_dim),
                                 input_resolution=(patches_resolution[0],
                                                   patches_resolution[1]),
                                 depth=depths[0],
                                 num_heads=num_heads[0],
                                 window_size=window_size,
                                 mlp_ratio=self.mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop_rate, attn_drop=attn_drop_rate,
                                 drop_path=dpr[sum(depths[:0]):sum(depths[:0 + 1])],
                                 norm_layer=norm_layer,
                                 use_checkpoint=use_checkpoint)
        self.EN2_0 = StageModule(dim=int(embed_dim),
                                 input_resolution=(patches_resolution[0],
                                                   patches_resolution[1]),
                                 depth=depths[1],
                                 num_heads=num_heads[1],
                                 window_size=window_size,
                                 mlp_ratio=self.mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop_rate, attn_drop=attn_drop_rate,
                                 drop_path=dpr[sum(depths[:1]):sum(depths[:1 + 1])],
                                 norm_layer=norm_layer,
                                 use_checkpoint=use_checkpoint)
        self.EN3_0 = StageModule(dim=int(embed_dim),
                                 input_resolution=(patches_resolution[0],
                                                   patches_resolution[1]),
                                 depth=depths[2],
                                 num_heads=num_heads[2],
                                 window_size=window_size,
                                 mlp_ratio=self.mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop_rate, attn_drop=attn_drop_rate,
                                 drop_path=dpr[sum(depths[:2]):sum(depths[:2 + 1])],
                                 norm_layer=norm_layer,
                                 use_checkpoint=use_checkpoint)

        #  decoder
        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(self.embed_dim * 2)

        self.conv = nn.Conv2d(in_channels=embed_dim, out_channels=self.out_chans, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

        self.prelu = nn.PReLU()
        self.tanh = nn.Tanh()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}
    
    # reshape&reorder
    def reshape_and_reorder(self, x):
        """
        Reshape and reorder the input tensor from (B, L, C) to (B, C, H, W) format.
        
        Returns:
            Tensor in shape (B, C, H, W)
        """
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 1, 2)
        return x

    # flatten&normalize
    def flatten_and_normalize(self, x):
        """
        Flatten the feature map and apply layer normalization.

        Returns:
            Flattened and normalized tensor of shape (B, L, C)
        """
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x

    # Encoder
    def encoder(self, x):
        """
        Encode the input image using stacked Swin Transformer blocks.

        Returns:
            A high-dimensional feature map of shape (B, C, H, W)
        """
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x1 = self.EN1_0(x)
        x1 = x1 + x
        x2 = self.EN2_0(x1)
        x2 = x2 + x1
        x3 = self.EN3_0(x2)
        x3 = x3 + x2
        x3 = self.norm(x3)
        x3 = self.reshape_and_reorder(x3)
        return x3

    # Dencoder
    def decoder(self, x):
        """
        Decode the encoded feature map to produce the output image.

        Returns:
            A single-channel reconstructed image.
        """
        x = self.conv(x)
        x = self.tanh(x)
        return x

    def autoencoder(self, x):
        """
        End-to-end pass through encoder and decoder modules.

        Returns:
            Output image after encoding and decoding.
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    @property
    def device(self):
        """
        Return the current device of the model parameters.
        """
        return next(self.parameters()).device
    
    def select_pixels(self,x,y):
        """
        Selects pixels from a 3D volume based on focus measure indices.

        Args:
            x (np.ndarray): Stack of color images (H, W, N, 3)
            y (np.ndarray): 2D focus index map (H, W)

        Returns:
            np.ndarray: Focused RGB image based on the index map.
        """
        w = x.shape[0]
        h = x.shape[1]
        x_tmp = x.reshape(w*h,*x.shape[2:])
        y_tmp = y.ravel()
        x_dest = np.array([xi[yi] for xi,yi in zip(x_tmp,y_tmp)])
        nshape = (w,h) if x_dest.ndim == 1 else (w,h,*x_dest.shape[1:])
        x_dest = x_dest.reshape(*nshape)
        return x_dest
    
    def fuse(self, images):
        """
        Fuse a sequence of defocused images into a single all-in-focus image 
        using the spatial frequency as focus measure.

        Args:
            images (list): List of image paths to process.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the estimated depth map and 
            the fused RGB image.
        """
        org_shape = np.array(Image.open(images[0]).convert('L')).shape
        outputs = []
        imgs_stack = []
        with tqdm(images, 
                  desc="Extracting Shape From Focus", 
                  bar_format="{desc}: {percentage:3.0f}%|{bar:20}| Images: {n}/{total}") as pbar:
            for image in pbar:
                img_org = Image.open(image).convert('RGB').resize((224, 224))         
                img = Image.open(image).convert('L').resize((224, 224))
                gimg = np.array(img)
                oimg = np.array(img_org)
                imgs_stack.append(oimg)
                imgn = np.expand_dims(np.expand_dims(gimg, axis=0), axis=1)
                imgt = torch.from_numpy(imgn.astype(np.float32)).to(self.device)

                with torch.no_grad():
                    output = self.encoder(imgt).clone().to('cpu')
                    outputs.append(output)
                del imgt  # Delete the tensor from the GPU to free up space.
                torch.cuda.empty_cache()
        
        combined_tensor = torch.concatenate(outputs, dim=0)
        dm = spatial_frequency(combined_tensor, self.device)       
        fused = self.select_pixels(np.stack(imgs_stack,axis=2),dm)

        dm = np.abs((dm - dm.max()))

        return dm, fused