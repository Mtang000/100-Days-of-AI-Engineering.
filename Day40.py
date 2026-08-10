import torch
import torch.nn as nn
import torch.nn.functional as F


embed_dim = 64
num_experts = 8
top_k = 2


word_token = torch.randn(1, embed_dim)


class Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.brain = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        return self.brain(x)


experts = nn.ModuleList([Expert() for _ in range(num_experts)])


router = nn.Linear(embed_dim, num_experts)

print("--- The Routing Process ---")


routing_logits = router(word_token)


routing_probs = F.softmax(routing_logits, dim=-1)


top_scores, top_indices = torch.topk(routing_probs, k=top_k, dim=-1)

print("Router evaluated all 8 Experts")
winner_1 = top_indices[0][0].item()
winner_2 = top_indices[0][1].item()

print(f"- Expert #{winner_1} (Confidence: {top_scores[0][0].item()*100:.1f}%)")
print(f"- Expert #{winner_2} (Confidence: {top_scores[0][1].item()*100:.1f}%)")


final_output = torch.zeros_like(word_token)


for i in range(top_k):
    expert_idx = top_indices[0][i].item()
    expert_weight = top_scores[0][i].item()

    expert_answer = experts[expert_idx](word_token)

    final_output += expert_weight * expert_answer

print("( The word was processed using only 25% of the total available brainpower. )")
