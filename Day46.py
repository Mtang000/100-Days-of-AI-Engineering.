import torch
import torch.nn as nn
import torch.nn.functional as F


vocab_size = 1000
embed_dim = 16
seq_length = 6

embedding = nn.Embedding(vocab_size, embed_dim)


combined_input = torch.tensor([[12, 45, 80, 55, 99, 102]])
print("Input sequence: [Query Words] + [Document Words]")


class CrossEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads=2, batch_first=True)

        self.scoring_layer = nn.Sequential(
            nn.Linear(embed_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x_embed):

        attn_out, _ = self.attention(x_embed, x_embed, x_embed)

        sentence_representation = attn_out[:, 0, :]

        score = self.scoring_layer(sentence_representation)
        return score


re_ranker = CrossEncoder()

embedded_input = embedding(combined_input)

print("\nRunning full self-attention on the combined text...")
match_score = re_ranker(embedded_input)

print(f"\nFinal Relevance Score: {match_score.item():.4f}")
print("-> (A score near 1.0 means perfect match, near 0.0 means terrible match)")
