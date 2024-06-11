import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
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
        self.patch_size = patch_size
        self.projection = nn.Sequential(
            # Break the image into patches
            nn.Conv2d(in_channels, emb_size, stride=stride, padding=padding, kernel_size=patch_size, stride=patch_size), # (B, C, H, W) -> (B, emb_size, (h - p)/s + 1, (w - p)/s + 1)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # num_patches = (h - p)/s + 1 * (w - p)/s + 1
        # numpatches_h = (h - p)/s + 1
        # numpatches_w = (w - p)/s + 1
        x = self.projection(x) # (B, C, H, W) -> (B, emb_size, num_patches_h, num_patches_w) 
        return x # (B, emb_size, num_patches_h, num_patches_w)

class Attention(nn.Module):
    def __init__(self, 
                 dim: int,
                 num_heads: int = 4,
                 qkv_bias: bool = False,
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
        B, H, W, C = x.shape
        
