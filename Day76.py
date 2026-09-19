import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class RewardModel(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, text_embedding):
        return self.scorer(text_embedding)


def bradley_terry_loss(chosen_scores, rejected_scores):
    score_difference = chosen_scores - rejected_scores

    loss = -F.logsigmoid(score_difference).mean()
    return loss


embed_dim = 128
reward_model = RewardModel(embed_dim)
optimizer = optim.Adam(reward_model.parameters(), lr=0.01)

chosen_answer = torch.randn(1, embed_dim)
rejected_answer = torch.randn(1, embed_dim)

for epoch in range(5):
    optimizer.zero_grad()

    chosen_score = reward_model(chosen_answer)
    rejected_score = reward_model(rejected_answer)

    loss = bradley_terry_loss(chosen_score, rejected_score)

    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1} | Chosen Score: {chosen_score.item():.4f} | Rejected Score: {rejected_score.item():.4f} | Loss: {loss.item():.4f}")
