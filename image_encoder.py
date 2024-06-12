import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from common import FeedFwd
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional



class PatchEmbedding(nn.Module):
    '''
        This function is responsible for creating patches of an image.
        We efficiently handle this process through CNN's.
        Using a kernel of patch size and similar stride we can generate x patches of an image.
        Then as we want to convert/map each patch to an embedding of D dimentions.
        We can do that by introducing D filters which generate D feature maps.
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
        # Combining the Batch and number of heads dimension, gives us a gist of multi-head attention.
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
        self.norm1 = norm_layer(dim)
        self.attn = CausalMultiHeadedAttention(dim, num_heads, qkv_bias=qkv_bias, input_shape=input_shape)
        self.norm2 = norm_layer(dim)
        self.mlp = FeedFwd(dim, int(dim * mlp_ratio), act_layer)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_fwd = x # Storing the input for residual connection. (B, H, W, dim)
        x = self.norm1(x) # Normalizing the input. As per paper

        x = self.attn(x) # Applying the attention mechanism. (B, H, W, dim)

        x = x + skip_fwd # Adding the residual connection. (B, H, W, dim) as per paper
        x = self.mlp(self.norm2(x)) # Applying the feed forward network. (B, H, W, dim) for giving time to the model to actually learn

        return x # (B, H, W, dim)