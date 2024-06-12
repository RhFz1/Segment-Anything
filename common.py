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
