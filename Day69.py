import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleBERTScore(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, cand_tokens, ref_tokens):
        cand_vecs = self.embedding(cand_tokens)
        ref_vecs = self.embedding(ref_tokens)
        cand_norm = F.normalize(cand_vecs, p=2, dim=-1)
        ref_norm = F.normalize(ref_vecs, p=2, dim=-1)

        sim_matrix = torch.matmul(cand_norm.squeeze(
            0), ref_norm.squeeze(0).transpose(0, 1))

        max_sim_cand, _ = torch.max(sim_matrix, dim=1)
        precision = torch.mean(max_sim_cand)
        max_sim_ref, _ = torch.max(sim_matrix, dim=0)
        recall = torch.mean(max_sim_ref)

        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

        return precision.item(), recall.item(), f1.item(), sim_matrix


vocab_size = 50
embed_dim = 16

model = SimpleBERTScore(vocab_size, embed_dim)

with torch.no_grad():
    model.embedding.weight[10] = model.embedding.weight[5]

cand_tokens = torch.tensor([[5]])
ref_tokens = torch.tensor([[10]])

p, r, f1, matrix = model(cand_tokens, ref_tokens)

print(
    f"Similarity Score between 'automobile' and 'car': {matrix[0, 0].item():.4f}")
print(f"BERTScore Precision : {p * 100:.1f}%")
print(f"BERTScore Recall    : {r * 100:.1f}%")
print(f"BERTScore F1        : {f1 * 100:.1f}%")
