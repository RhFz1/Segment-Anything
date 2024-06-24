import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from common import FeedFwd
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional, Type


class ImageEncoder(nn.Module):

    def __init__(self,
                 img_size: int = 1024,
                 patch_size: int = 16, 
                 emb_size: int = 768,
                 in_channels: int = 3,
                 num_layers: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 out_chans: int = 256,
                 qkv_bias: bool = False,
                 norm_layer: Type[nn.Module] = nn.LayerNorm,
                 act_layer: Type[nn.Module] = nn.GELU,
                 use_abs_pos: bool = True,
                 use_rel_pos: bool = False,
                 rel_pos_zero_init: bool = True,
                 window_size: int = 0,
            ) -> None:
        super().__init__()

        self.img_size = img_size

        self.patch_embd = PatchEmbedding(
            in_channels=in_channels,
            patch_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            emb_size=emb_size,
            img_size=(img_size, img_size)
        )

        self.pos_embd: Optional[nn.Parameter] = None

        if use_abs_pos:
            self.pos_embd = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, emb_size) # (1, H, W, emb_size)
            )
        
        self.blocks = nn.ModuleList()

        for _ in range(num_layers):
            self.blocks.append(
                TransformerBlock(
                    dim=emb_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    act_layer=act_layer
                )
            )
        
        # Here this block is the neck of the image encoder.
        # We try to take the output of the transformer block and then pass it through a series of convolutional layers.
        # This is done to ensure the model is robust to the spatial information.
        # (B, H, W, emb_size) -> (B, H, W, out_chans) 
        # H = (h + 2p - k)/s + 1, W = (w + 2p - k)/s + 1
        
        self.neck = nn.Sequential(
            nn.Conv2d(emb_size, out_chans, kernel_size=1), # Here conv. operation is used to reduce the number of channels.
            nn.LayerNorm(out_chans),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False), # Based on the paper, we use 3x3 kernel size and padding of 1. Which retains the map shapes.
            nn.LayerNorm(out_chans)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embd(x) # Conversion of image to patches. (B, H, W, C) -> (B, num_patches_h, num_patches_w, emb_size)

        # Paper explains the use of relative position embeddings. To preserve the relative positioning of patch content.
        if self.pos_embd is not None:
            x = x + self.pos_embd
        
        # Passing the patches through the transformer blocks.
        for block in self.blocks:
            x = block(x)

        # Permuting here as torch expects the input to be in the format of (B, C, H, W)
        x = self.neck(x.permute(0, 3, 1, 2)) # (B, num_patches_h, num_patches_w, emb_size) -> (B, out_chans, num_patches_h, num_patches_w)

        return x
class PatchEmbedding(nn.Module):
    '''
        This function is responsible for creating patches of an image.
        We efficiently handle this process through CNN's.
        Using a kernel of patch size and similar stride we can generate x patches of an image.
        Then as we want to convert/map each patch to an embedding of D dimentions.
        We can do that by introducing D filters which generate D feature maps.
        Each pixel of the feature map is the embedding of the corresponding patch, i.e., if we have a 64x64 map
        Then each pixel of the feature map is the embedding of the corresponding 16x16 patch.
    '''
    def __init__(self,
                 in_channels: int = 3,
                 patch_size: Tuple[int, int] = (16, 16),
                 stride: Tuple[int, int] = (16, 16),
                 padding: Tuple[int, int] = (0, 0),
                 emb_size: int = 768, 
                 img_size: Tuple[int, int] = (224, 224)) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Sequential(
            # Break the image into patches
            nn.Conv2d(in_channels, emb_size, stride=stride, padding=padding, kernel_size=patch_size), # (B, C, H, W) -> (B, emb_size, (h - p)/s + 1, (w - p)/s + 1)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # num_patches = (h - p)/s + 1 * (w - p)/s + 1
        # numpatches_h = (h - p)/s + 1
        # numpatches_w = (w - p)/s + 1
        x = self.projection(x) # (B, C, H, W) -> (B, emb_size, num_patches_h, num_patches_w) 
        x = x.permute(0, 2, 3, 1) # (B, num_patches_h, num_patches_w, emb_size)
        return x # (B, num_patches_h, num_patches_w, emb_size)

class CausalMultiHeadedAttention(nn.Module):
    """
        This function is responsible for calculating the attention scores.
        Here we are trying to implement the multi-head attention mechanism, which is the core of the transformer model.
        Firstly we aggregate the Query, Key and Value into a single tensor.
        Compute q,k,v in parallel. For all heads and we basically reshape them accordingly.
        Then send them out through projection layers.
    """
    def __init__(self, 
                 dim: int,
                 num_heads: int = 4,
                 qkv_bias: bool = False,
                 use_rel_pos: bool = False,
                 rel_pos_zero_init: bool = True,
                 input_shape: Optional[Tuple[int , int]] = None) -> None:
        super().__init__()


        self.num_heads = num_heads # The number of heads in a single attention layer or the number of parallel attention layers. (Multi-Head Attention)
        self.head_dim = dim // num_heads # The dimension of each head it is the split of what each head will see.
        self.scale = self.head_dim ** -0.5 # The scaling factor for the dot product attention. Refer to the paper for more details.

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) # The linear layer for the Query, Key and Value, all combined into one. Will be separated later.
        self.att_drop = nn.Dropout(0.1)
        self.projection = nn.Linear(dim, dim)
        self.input_shape = input_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape

        # Here we are trying to implement the multi-head attention mechanism.
        # The multiplier 3 corresponds to the Query, Key and Value.
        # We know each head shall process (emb_size/num_heads) features.
        # So we reshape the output of the linear layer to (B, H * W, 3, num_heads, head_dim)
        # Then we permute the tensor to (3, B, num_heads, H * W, head_dim)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)

        # here we are trying to separate the Query, Key and Value.
        # Combining the Batch and number of heads dimension, gives us a gist of multi-heemb_sizead attention.
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, self.head_dim).unbind(0) # (3, B * num_head, H * W, head_dim)

        # Calculating the attention scores.
        wei = (q * self.scale) @ (k.transpose(-2, -1)) # (B * num_head, H * W, head_dim) * (B * num_head, head_dim, H * W) -> (B * num_head, H * W, H * W)
        wei = F.softmax(wei, dim=-1) # Normalizing the attention scores.
        wei = self.att_drop(wei)

        # Calculating the weighted sum of the values.
        out = (wei @ v) # (B * num_head, H * W, H * W) * (B * num_head, H * W, head_dim) -> (B * num_head, H * W, head_dim)
        out = out.view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4) #(B, num_head, H, W, head_dim) -> (B, H, W, num_head, head_dim)
        proj = self.projection(out.reshape(B, H , W, -1)) # (B, H, W, dim)
        return proj

class TransformerBlock(nn.Module):
    """
        Here this block is the combination of all the components of the transformer model.
        It contains the normalization, followed by attention, then again normalization and then the feed forward network.
        Still have to implement the positional embedding modules, to ensure each token is aware of its position and relative position wrto others.
    """

    def __init__(self,
                 dim: int = 768,
                 num_heads: int = 4,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = False,
                 norm_layer: nn.Module = nn.LayerNorm,
                 act_layer: nn.Module = nn.GELU,
                 input_shape: Optional[Tuple[int, int]] = None
                 ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim) # Layer Normalization for ensuring the model is robust to the scale of the input.
        self.attn = CausalMultiHeadedAttention(dim, num_heads, qkv_bias=qkv_bias, input_shape=input_shape) # The attention mechanism.
        self.norm2 = norm_layer(dim) # Normalization layer for the output of the attention mechanism.
        self.mlp = FeedFwd(dim, int(dim * mlp_ratio), act_layer) # The feed forward network.
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Here we keep a copy of the input for the residual connection.
        # Then as per paper we first normalize the input x.
        # Then we apply the attention mechanism, where in the model shall learn the sequential dependencies.
        # Then we add the residual connection, followeed by the feed forward network.

        skip_fwd = x # Storing the input for residual connection. (B, H, W, dim)
        x = self.norm1(x) # Normalizing the input. As per paper

        x = self.attn(x) # Applying the attention mechanism. (B, H, W, dim)

        x = x + skip_fwd # Adding the residual connection. (B, H, W, dim) as per paper
        x = self.mlp(self.norm2(x)) # Applying the feed forward network. (B, H, W, dim) for giving time to the model to actually learn

        return x # (B, H, W, dim)   
    

# Still do not understand the motivation behind the window partition.
# Will try to explain here when understood the justification of use.

def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
        This function is responsible for partitioning the image into windows.
        We are trying to break the image into windows of size window_size.
        The image is broken into windows of size window_size x window_size.
    """
    B, H, W, C = x.shape
    

    pad_h = (window_size - H % window_size) % window_size # Padding the height of the image.
    pad_w = (window_size - W % window_size) % window_size # Padding the width of the image.

    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0 , 0, pad_w, 0, pad_h), value=0) # Padding the image.
    
    Hn, Wn = H + pad_h, W + pad_w # New height and width of the image.
    x = x.view(B, Hn // window_size,window_size ,Wn // window_size,window_size,C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C) # (B * Hn//window_size * Wn//window_size, window_size, window_size, C)
    
    return x, (Hn, Wn)

def window_unpartition(x: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]) -> torch.Tensor:
    """
        This function is responsible for unpartitioning the image into windows.
        This function is the inverse of the window_partition function.
    """
    Hn, Wn = pad_hw[0], pad_hw[1]
    H, W = hw[0], hw[1]

    # Computing the batch size.
    B = x.shape[0] // (Hn * Wn // window_size ** 2)

    x = x.view(B, Hn // window_size, Wn// window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hn, Wn, -1)

    if Hn > H or Wn > W:
        x = x[:, :H, :W, :]
    
    return x

def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:

    """
        This function gives out the relative position of the query and key to ensure attention is not disturbed.
        When the image contents are displaced and patch contents are different altogether.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]