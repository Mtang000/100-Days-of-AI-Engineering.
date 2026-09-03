import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)

        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)

        self.embed_dim = embed_dim

    def forward(self, decoder_state, encoder_transcript):
        Q = self.W_q(decoder_state)
        K = self.W_k(encoder_transcript)
        V = self.W_v(encoder_transcript)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1))
        attention_scores = attention_scores / (self.embed_dim ** 0.5)

        attention_weights = F.softmax(attention_scores, dim=-1)

        summary_context = torch.matmul(attention_weights, V)

        return summary_context, attention_weights


embed_size = 128
seq2seq_attention = CrossAttention(embed_dim=embed_size)

transcript_tensors = torch.randn(1, 500, embed_size)

current_summary_state = torch.randn(1, 1, embed_size)

print(f"Transcript Matrix Shape: {transcript_tensors.shape} (500 Words)")
print(f"Decoder Target Shape: {current_summary_state.shape} (1 Word)")
print("-" * 50)

context_vector, heatmap_weights = seq2seq_attention(
    current_summary_state, transcript_tensors)

print(f"Filtered Summary Context Shape: {context_vector.shape}")

highest_score_idx = torch.argmax(heatmap_weights).item()
highest_score_val = heatmap_weights[0, 0, highest_score_idx].item() * 100

print(
    f"Out of 500 words, the Decoder decided that Word #{highest_score_idx} in the transcript was the absolute most critical, assigning it {highest_score_val:.2f}% of its total attention.")
