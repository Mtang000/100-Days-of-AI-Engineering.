import torch
import torch.nn.functional as F


def dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=0.1):

    policy_logratios = policy_chosen_logps - policy_rejected_logps

    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = policy_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits).mean()

    return loss


reference_chosen_logp = torch.tensor([-2.5])
reference_rejected_logp = torch.tensor([-2.1])
policy_chosen_logp_step1 = torch.tensor([-2.5])
policy_rejected_logp_step1 = torch.tensor([-2.1])

initial_loss = dpo_loss(
    policy_chosen_logp_step1,
    policy_rejected_logp_step1,
    reference_chosen_logp,
    reference_rejected_logp,
    beta=0.1
)

print(f"\nTraining Step 1 Loss: {initial_loss.item():.4f}")

policy_chosen_logp_step50 = torch.tensor([-1.2])
policy_rejected_logp_step50 = torch.tensor([-4.5])

# Calculate optimized loss
optimized_loss = dpo_loss(
    policy_chosen_logp_step50,
    policy_rejected_logp_step50,
    reference_chosen_logp,
    reference_rejected_logp,
    beta=0.1
)

print(f"Training Step 50 Loss: {optimized_loss.item():.4f}")
