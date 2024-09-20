import torch
from torch import nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.sqrt_d_k = torch.sqrt(self.d_k)
        self.softmax = nn.Softmax(dim=1)
        
        
    def forward(self, Q_head: torch.Tensor, K_head: torch.Tensor, V_head: torch.Tensor, mask: torch.Tensor):
        
        # batch_size, seq_len, heads, d_model
        numerator = torch.matmul(Q_head, K_head.transpose(2,3))
        q_k_tmp :torch.Tensor = numerator/self.sqrt_d_k
        q_k_tmp = q_k_tmp.data.masked_fill_(mask, -torch.finfo(torch.float).max)
        q_k_tmp = self.softmax(q_k_tmp)
        
        attention = torch.matmul(q_k_tmp,V_head)
        return attention
        