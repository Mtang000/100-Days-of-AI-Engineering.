import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAttention(nn.Module):
    def __init__(self, embed_dim, window_size):
        super().__init__()
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)

        self.embed_dim = embed_dim
        self.window_size = window_size

    def create_sliding_window_mask(self, sequence_length):
        mask = torch.zeros(sequence_length, sequence_length, dtype=torch.bool)
        for i in range(sequence_length):
            start = max(0, i - self.window_size)
            end = min(sequence_length, i + self.window_size + 1)
            mask[i, start:end] = True
        return mask

    def forward(self, input_sequence):
        """
        input_sequence: Tensor of shape (batch_size, sequence_length, embed_dim)
        """
        sequence_length = input_sequence.shape[1]
        Q = self.W_q(input_sequence)
        K = self.W_k(input_sequence)
        V = self.W_v(input_sequence)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.embed_dim ** 0.5)

        sparse_mask = self.create_sliding_window_mask(sequence_length)

        scores = scores.masked_fill(~sparse_mask, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)

        context_vector = torch.matmul(attention_weights, V)

        return context_vector, attention_weights


embed_dim = 64
sequence_length = 20
window_size = 2

layer = SparseAttention(embed_dim=embed_dim, window_size=window_size)

document_data = torch.randn(1, sequence_length, embed_dim)

context, weights = layer(document_data)

print(f"Sequence Length: {sequence_length} words")
print(f"Window Size: +/- {window_size} words")
print(f"Output Context Matrix Shape: {context.shape}")

print("\nAttention Matrix Row 5 (Word 5):")
row_scores = weights[0, 5].tolist()
formatted_scores = [f"{score:.2f}" for score in row_scores]
print(formatted_scores)
