import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class TwoTowerModel(nn.Module):
    def __init__(self, vocab_size=1000, embed_dim=16):
        super().__init__()

        self.query_embedding = nn.Embedding(vocab_size, embed_dim)
        self.query_network = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, embed_dim)
        )

        self.doc_embedding = nn.Embedding(vocab_size, embed_dim)
        self.doc_network = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, embed_dim)
        )

    def forward(self, query, positive_doc, negative_doc):

        q_embed = self.query_embedding(query).mean(dim=1)
        q_vector = self.query_network(q_embed)

        pos_embed = self.doc_embedding(positive_doc).mean(dim=1)
        pos_vector = self.doc_network(pos_embed)

        neg_embed = self.doc_embedding(negative_doc).mean(dim=1)
        neg_vector = self.doc_network(neg_embed)

        q_vector = F.normalize(q_vector, p=2, dim=1)
        pos_vector = F.normalize(pos_vector, p=2, dim=1)
        neg_vector = F.normalize(neg_vector, p=2, dim=1)

        return q_vector, pos_vector, neg_vector


model = TwoTowerModel()
optimizer = optim.Adam(model.parameters(), lr=0.01)

query_data = torch.tensor([[12, 45, 89]])

pos_doc_data = torch.tensor([[55, 99, 102]])


neg_doc_data = torch.tensor([[201, 305, 410]])

model.eval()
with torch.no_grad():
    q_vec, p_vec, n_vec = model(query_data, pos_doc_data, neg_doc_data)
    pos_sim_before = F.cosine_similarity(q_vec, p_vec).item()
    neg_sim_before = F.cosine_similarity(q_vec, n_vec).item()

print("--- BEFORE TRAINING (Random Weights) ---")
print(f"Similarity (Query <-> Good Doc):  {pos_sim_before:.4f}")
print(f"Similarity (Query <-> Bad Doc):   {neg_sim_before:.4f}\n")


model.train()

for epoch in range(10):
    optimizer.zero_grad()

    q_vec, p_vec, n_vec = model(query_data, pos_doc_data, neg_doc_data)

    pos_similarity = torch.sum(q_vec * p_vec, dim=-1)
    neg_similarity = torch.sum(q_vec * n_vec, dim=-1)

    loss = - (pos_similarity - neg_similarity).mean()

    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    q_vec, p_vec, n_vec = model(query_data, pos_doc_data, neg_doc_data)
    pos_sim_after = F.cosine_similarity(q_vec, p_vec).item()
    neg_sim_after = F.cosine_similarity(q_vec, n_vec).item()

print("\n--- AFTER 10 STEPS OF TRAINING ---")
print(
    f"Similarity (Query <-> Good Doc):  {pos_sim_after:.4f} (PULLED CLOSER!)")
print(f"Similarity (Query <-> Bad Doc):   {neg_sim_after:.4f} (PUSHED AWAY!)")
