import torch
import torch.nn.functional as F

print("--- Top-K & Top-P ---\n")


vocabulary = [
    "keyboard",
    "switch",
    "tape",
    "door",
    "software",
    "potato",     # - Gibberish!
    "flying",     # - Gibberish!
]

logits = torch.tensor([8.0, 6.0, 4.0, 2.0, 1.0, 0.5, 0.1])


def apply_top_k(logits, k=3):
    """Keeps only the Top K highest scores, changes the rest to -Infinity."""
    top_k_values, _ = torch.topk(logits, k)
    minimum_allowed_score = top_k_values[-1]

    filtered_logits = torch.where(
        logits < minimum_allowed_score,
        torch.tensor(float('-inf')),
        logits
    )
    return filtered_logits


def apply_top_p(logits, p=0.90):
    """Keeps words until their combined percentages hit the P target."""

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)

    sorted_probs = F.softmax(sorted_logits, dim=0)

    cumulative_probs = torch.cumsum(sorted_probs, dim=0)

    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
    sorted_indices_to_remove[0] = 0

    indices_to_remove = sorted_indices_to_remove.scatter(
        0, sorted_indices, sorted_indices_to_remove)

    filtered_logits = logits.masked_fill(indices_to_remove, float('-inf'))
    return filtered_logits


print("1. NO FILTERS (High Risk)")
probs = F.softmax(logits, dim=0)
print(f"Words available to pick: {len(vocabulary)}")
print(f"Chance of picking 'potato': {probs[5].item()*100:.1f}%\n")

print("2. TOP-K FILTER (k=3)")
top_k_logits = apply_top_k(logits, k=3)
top_k_probs = F.softmax(top_k_logits, dim=0)
print(f"Words available: 3 (The rest are blocked)")
print(f"Chance of picking 'potato': {top_k_probs[5].item()*100:.1f}%\n")

print("3. TOP-P (NUCLEUS) FILTER (p=0.80)")
top_p_logits = apply_top_p(logits, p=0.80)
top_p_probs = F.softmax(top_p_logits, dim=0)
print(f"The AI added up percentages until it hit 80%, then blocked the rest.")
print(f"Chance of picking 'potato': {top_p_probs[5].item()*100:.1f}%\n")
