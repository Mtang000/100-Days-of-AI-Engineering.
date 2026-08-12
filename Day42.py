import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class TinyLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.brain = nn.Linear(10, 5)

    def forward(self, x):
        return self.brain(x)


policy_model = TinyLLM()


reference_model = TinyLLM()
reference_model.load_state_dict(policy_model.state_dict())
reference_model.requires_grad_(False)  # Locked!

optimizer = optim.Adam(policy_model.parameters(), lr=0.1)


context = torch.randn(1, 10)
chosen_word_idx = torch.tensor([2])
rejected_word_idx = torch.tensor([4])


print("--- Before DPO Training ---")
logits = policy_model(context)
probs = torch.softmax(logits, dim=-1)
print(
    f"Chosen (Polite) Probability:  {probs[0][chosen_word_idx].item()*100:.2f}%")
print(
    f"Rejected (Toxic) Probability: {probs[0][rejected_word_idx].item()*100:.2f}%\n")


policy_logits = policy_model(context)
policy_log_probs = F.log_softmax(policy_logits, dim=-1)
policy_chosen_log_prob = policy_log_probs[0, chosen_word_idx]
policy_rejected_log_prob = policy_log_probs[0, rejected_word_idx]

with torch.no_grad():
    ref_logits = reference_model(context)
    ref_log_probs = F.log_softmax(ref_logits, dim=-1)
    ref_chosen_log_prob = ref_log_probs[0, chosen_word_idx]
    ref_rejected_log_prob = ref_log_probs[0, rejected_word_idx]


beta = 0.1

chosen_ratio = policy_chosen_log_prob - ref_chosen_log_prob
rejected_ratio = policy_rejected_log_prob - ref_rejected_log_prob


logits_diff = chosen_ratio - rejected_ratio
dpo_loss = -F.logsigmoid(beta * logits_diff)


optimizer.zero_grad()
dpo_loss.backward()
optimizer.step()

print("--- After 1 Step of DPO Training ---")
new_logits = policy_model(context)
new_probs = torch.softmax(new_logits, dim=-1)
print(
    f"Chosen (Polite) Probability:  {new_probs[0][chosen_word_idx].item()*100:.2f}% (Increased)")
print(
    f"Rejected (Toxic) Probability: {new_probs[0][rejected_word_idx].item()*100:.2f}% (Decreased)")
