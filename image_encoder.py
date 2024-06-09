import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, List, Dict



class PatchEmbedding(nn.Module):
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
            # Flatten the image into a sequence of patches (H / patch_size) * (W / patch_size) = num_patches
            nn.Flatten(start_dim=2, end_dim=3) # (B, emb_size, H, W) -> (B, emb_size, num_patches)
        )
    def forward(self, x):
        x = self.projection(x) # (B, C, H, W) -> (B, emb_size, num_patches)
        return x # (B, emb_size, num_patches)