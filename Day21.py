import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_length=100):
        super().__init__()

        pe = torch.zeros(max_length, embed_dim)

        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(
            0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term)

        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):

        seq_length = x.size(1)

        x = x + self.pe[:, :seq_length, :]
        return x


embed_dim = 4
sentence_length = 3
batch_size = 1


blank_embeddings = torch.zeros(batch_size, sentence_length, embed_dim)

print("Original Embeddings (Before Positional Encoding):")
print(blank_embeddings[0])
print("-" * 50)

pos_encoder = PositionalEncoding(embed_dim=embed_dim)

time_stamped_embeddings = pos_encoder(blank_embeddings)

print("Time-Stamped Embeddings (After Positional Encoding):")
print(time_stamped_embeddings[0])
