import torch
from torch import nn


class WordEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int, #number of unique words
        embedding_dim: int, #number of features to embed the word to
        pad_idx: int, #index of the pad feature
    ):
        super(WordEmbedding, self).__init__()
        self.embeder = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim, padding_idx=pad_idx)
    
    def forward(self, input: torch.Tensor):
        return self.embeder(input)