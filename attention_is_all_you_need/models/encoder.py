import torch
from torch import nn
from models.multiheadattention import MultiHeadedAttention
from models.wordembedding import WordEmbedding
from models.positionencoder import PositionEncoder
from models.ff import FF

class EncoderLayer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_idx: int,
        max_seq_len: int,
        d_model: int = 512,
        ff_dim: int = 2048,
        heads: int = 8,
    ):
        super(EncoderLayer, self).__init__()
        
        self.multiheadedattention = MultiHeadedAttention(heads=heads,d_model=d_model)
        self.FF = FF(d_model=d_model, ff_dim=ff_dim)
        
    def forward(self, input_batch : torch.Tensor, mask : torch.Tensor):
        x = self.multiheadedattention(x,x,x,mask)
        x = self.FF(x)
        

class Encoder(nn.Module):
    
    def __init__(
        self,
        mask : torch.Tensor,
        N_layers : torch.Tensor,
        vocab_size: int,
        pad_idx: int,
        max_seq_len: int,
        d_model: int = 512,
        ff_dim: int = 2048,
        heads: int = 8,
    ):
        super(Encoder, self).__init__()
        
        self.embedder = WordEmbedding(num_embeddings=vocab_size, embedding_dim=d_model, pad_idx=pad_idx)
        self.positionencoder = PositionEncoder(d_model=d_model, max_seq_len=max_seq_len)
        
        
        self.N_layers = N_layers
        self.mask = mask
        
        self._EncoderLayersBuilder = []
        
        for layer in range(N_layers):
            self._EncoderLayersBuilder.append(
                EncoderLayer(
                    vocab_size=vocab_size,
                    pad_idx=pad_idx,
                    max_seq_len=max_seq_len,
                    d_model=d_model,
                    ff_dim=ff_dim,
                    heads=heads
                )
            )
        
        self.EncoderLayers = nn.Sequential(*self._EncoderLayersBuilder)
        
    def forward(self, input_batch : torch.Tensor, mask :torch.Tensor):
        x = self.embedder(input_batch)
        x = self.positionencoder(x)
        x = self.EncoderLayers(x)
        
        return x
    