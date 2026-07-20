import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):

        residual_1 = x

        attn_output, _ = self.attention(query=x, key=x, value=x)

        x = self.norm1(attn_output + residual_1)

        residual_2 = x

        ff_output = self.feed_forward(x)

        out = self.norm2(ff_output + residual_2)

        return out


batch_size = 1
sequence_length = 5
embed_dim = 16
num_heads = 4

dummy_sentence = torch.rand(batch_size, sequence_length, embed_dim)

gpt_block = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads)

output = gpt_block(dummy_sentence)

print(f"Input shape (Batch, Words, Vector Size): {dummy_sentence.shape}")
print(f"Output shape (Batch, Words, Vector Size): {output.shape}\n")
print("Success! The data flowed through Attention, Feed-Forward, and Normalization perfectly.")
