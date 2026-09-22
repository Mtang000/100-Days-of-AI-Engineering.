import torch
import torch.nn as nn
import torch.nn.functional as F


class MixtureOfExperts(nn.Module):
    def __init__(self, embed_dim, num_experts, active_experts):
        super().__init__()
        self.num_experts = num_experts
        self.active_experts = active_experts

        self.router = nn.Linear(embed_dim, num_experts, bias=False)

        self.experts = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(num_experts)
        ])

    def forward(self, x):
        routing_logits = self.router(x)
        routing_probabilities = F.softmax(routing_logits, dim=-1)

        top_probabilities, top_indices = torch.topk(
            routing_probabilities,
            self.active_experts,
            dim=-1
        )

        final_output = torch.zeros_like(x)

        for i in range(self.active_experts):
            expert_index = top_indices[0, i].item()
            expert_weight = top_probabilities[0, i].item()

            print(
                f"-> Routing to Expert #{expert_index} (Weight: {expert_weight:.2f})")

            expert_result = self.experts[expert_index](x)

            final_output += (expert_weight * expert_result)

        return final_output


embed_dim = 128
total_experts = 8
experts_to_use = 2

moe_layer = MixtureOfExperts(
    embed_dim, num_experts=total_experts, active_experts=experts_to_use)

print("Input: A single word vector.")
word_vector = torch.randn(1, embed_dim)

with torch.no_grad():
    moe_layer.router.weight[3] = word_vector.squeeze() * 5.0
    moe_layer.router.weight[7] = word_vector.squeeze() * 3.0

output_vector = moe_layer(word_vector)
