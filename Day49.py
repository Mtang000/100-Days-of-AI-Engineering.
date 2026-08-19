import torch
import torch.nn.functional as F


vocab = ["apple", "banana", "orange", "shoe", "truck"]
raw_logits = torch.tensor([5.0, 4.0, 3.0, -1.0, -2.0])

print("--- Raw Probabilities (No Sampling Tricks) ---")
normal_probs = F.softmax(raw_logits, dim=-1)
for i in range(len(vocab)):
    print(f"- {vocab[i]:<10}: {normal_probs[i].item()*100:>5.1f}%")


temperature = 1.5
temp_logits = raw_logits / temperature

print(f"\n--- After High Temperature (T={temperature}) ---")
temp_probs = F.softmax(temp_logits, dim=-1)
for i in range(len(vocab)):
    print(f"- {vocab[i]:<10}: {temp_probs[i].item()*100:>5.1f}%")


sorted_probs, sorted_indices = torch.sort(temp_probs, descending=True)

cumulative_probs = torch.cumsum(sorted_probs, dim=-1)


top_p = 0.90

sorted_indices_to_remove = cumulative_probs > top_p
sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
sorted_indices_to_remove[..., 0] = 0

indices_to_remove = sorted_indices[sorted_indices_to_remove]
final_logits = temp_logits.clone()
final_logits[indices_to_remove] = float('-inf')

print(f"\n--- After Top-P Filter (P={top_p}) ---")
final_probs = F.softmax(final_logits, dim=-1)
for i in range(len(vocab)):
    if final_probs[i] > 0:
        print(f"- {vocab[i]:<10}: {final_probs[i].item()*100:>5.1f}%")
    else:
        print(f"- {vocab[i]:<10}: BANNED BY TOP-P")
