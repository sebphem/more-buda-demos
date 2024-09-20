import torch
from torch import nn
import numpy as np

class PositionEncoder(nn.Module):
    def __init__(self, d_model: int, max_seq_len:int):
        super(PositionEncoder, self).__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.sin = np.sin
        self.cos = np.cos
        # [all_positions, all_encoding_dimensions]
        self.register_buffer("positional_encodings",torch.zeros(max_seq_len,d_model))
        self._generate_positional_encodings_matrix()
        self.softmax = nn.Softmax(dim=1)
        
    
    def _calculate_encoding(self, k : int, i : int):
        # https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/
        # k = position in sequence
        # i = position in embedding
        freq = 1/10000*((2*i/self.d_model))
        x = k * freq
        # if odd
        if i % 2:
            # positional encoding
            pe = self.cos(x)
        # if even
        else:
            pe = self.sin(x)
        return pe
    
    def _generate_positional_encodings_matrix(self,):
        # shift up by 1 otherwise the first token just gets phased out
        for k in range(self.max_seq_len+1):
            for i in range(self.d_model+1):
                self.positional_encodings[k][i] = self._calculate_encoding(k,i)    
    
    def forward(self, input_embeddings: torch.Tensor):
        # need to do this dynamically to account for variable seq length for output
        batch_size, seq_len, d_model = input_embeddings.size(0), input_embeddings.size(1), input_embeddings.size(2)
        
        # z, seq, d_model + 1, seq, d_model makes the second tensor apply to all batches
        return input_embeddings + self.positional_encodings[:seq_len,:].unsqueeze(1)
    


if __name__ == "__main__":
    pe = PositionEncoder(d_model=100, max_seq_len=8)
    print(pe.positional_encodings)
    