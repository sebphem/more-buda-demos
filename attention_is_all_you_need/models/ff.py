import torch
from torch import nn
from torch.nn import ReLU

class FF(nn.Module):
    def __init__(self, d_model: int = 512, ff_dim: int = 2048):
        super(FF, self).__init__()
        self.l1 = nn.Linear(d_model, ff_dim, bias=True)
        self.l2 = nn.Linear(ff_dim, d_model, bias=True)
        self.relu = nn.ReLU()
    
    def forward(self, input: torch.Tensor):
        x = self.l1(input)
        x = self.relu(x)
        x = self.l2(x)
        return x