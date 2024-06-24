import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedFwd(nn.Module):
    """
        Here implementing Feed Forward Neural Network.
        Kept here in commons as it is going to be used indefinitely.
        Reason for having ffwd network is for the transformer to actually what it has learnt by performing attention.
    """

    def __init__(self,
                 dim: int = 768,
                 mlp_dim: int = 3072,
                 act_layer: nn.Module = nn.GELU) -> None:
        super().__init__()
        self.ln1 = nn.Linear(dim, mlp_dim)
        self.ln2 = nn.Linear(mlp_dim, dim)
        self.act = act_layer()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln2(self.act(self.ln1(x)))
        return x

class LayerNorm2d(nn.Module):
    """
        Implementing this to serve a specific case in the neck of the transformer, where we need to normalize the channel dimension.
        LayerNorm is used to normalize the input tensor on a specific dimension, which is the channel dimension in this case.
        It has proven to be better than BatchNorm in many cases, as it does not add the noise that BatchNorm does for a batch.
    """

    def __init__(self,
                 num_channels:int = 256,
                 eps: float = 1e-6) -> None:
        super().__init__()
        self.ln_gain = nn.Parameter(torch.ones(num_channels))
        self.ln_bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.view(B, C, -1)
        x = (x - x.mean(dim=1, keepdim=True)) * torch.rsqrt(x.var(dim=1, keepdim=True) + self.eps)  # Adding epsilon to avoid division by zero
       
        x = x.view(B, C, H, W) # Reshaping back to the original shape

        return x # Normalized tensor