from attention import ScaledDotProductAttention
from torch import nn
import torch

class MultiHeadedAttention(nn.Module):
    def __init__(self, heads:int = 8, d_model:int = 512):
        super(MultiHeadedAttention, self).__init__()
        self.heads = heads
        self.d_q = d_model // heads
        self.d_k = d_model // heads
        self.d_v = d_model // heads
        
        self.w_q = nn.Parameter(
            torch.Tensor(
                heads,d_model,self.d_k
            )
        )
        self.w_k = nn.Parameter(
            torch.Tensor(
                heads,d_model,self.d_k
            )
        )
        self.w_v = nn.Parameter(
            torch.Tensor(
                heads,d_model,self.d_k
            )
        )
        
        self.scaleddotproductattention = ScaledDotProductAttention(self.d_k)
        
        self.l1 = nn.Linear(heads * self.d_v, d_model, bias=False)
        
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor):
        
        # need to do this dynamically to account for variable seq length for output
        batch_size, seq_len, d_model = Q.size(0), Q.size(1), Q.size(2)
        
        # batch_size, seq_len, d_model -> batch_size, seq_len, heads, d_model
        Q = torch.matmul(self.w_q, Q)
        Q = Q.unsqueeze(2)
        Q = Q.repeat(1,1,self.heads,1)
        K = torch.matmul(self.w_k, K)
        K = K.unsqueeze(2)
        K = K.repeat(1,1,self.heads,1)
        V = torch.matmul(self.w_v, V)
        V = V.unsqueeze(2)
        V = V.repeat(1,1,self.heads,1)
        
        # hopefully this works
        attention_out : torch.Tensor = self.scaleddotproductattention(Q,K,V,mask)
        
        # concat
        attention_out = attention_out.view(batch_size,seq_len,-1)
        
        # multiply by weights matrix
        out = self.l1(attention_out)
        
        return out