import torch
import torch.nn as nn
import torch.nn.functional as F


sentence = ["the", "bank", "of", "the", "river"]


embeddings = {
    "the":   torch.tensor([[0.1, 0.2, 0.1, 0.1]]),
    "bank":  torch.tensor([[0.9, 0.1, 0.1, 0.8]]),
    "of":    torch.tensor([[0.1, 0.1, 0.1, 0.2]]),
    "river": torch.tensor([[0.8, 0.2, 0.1, 0.9]])
}

sequence_matrix = torch.cat([embeddings[word] for word in sentence])


class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.query_layer = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_layer = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_layer = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):

        Q = self.query_layer(x)
        K = self.key_layer(x)
        V = self.value_layer(x)

        attention_scores = torch.matmul(Q, K.transpose(0, 1))

        d_k = Q.size(-1)
        scaled_scores = attention_scores / (d_k ** 0.5)

        attention_weights = F.softmax(scaled_scores, dim=-1)

        context_vector = torch.matmul(attention_weights, V)

        return context_vector, attention_weights


torch.manual_seed(42)

attention_module = SelfAttention(embed_dim=4)


with torch.no_grad():
    final_output, weights = attention_module(sequence_matrix)

print(f"Sentence: {' | '.join(sentence)}\n")
print("Attention Weights Matrix (Percentages):")
print("-" * 50)

for i, word1 in enumerate(sentence):
    row_str = f"{word1:>6} sees: "
    for j, word2 in enumerate(sentence):

        percentage = weights[i, j].item() * 100
        row_str += f"{percentage:>4.1f}% ({word2})  "
    print(row_str)
