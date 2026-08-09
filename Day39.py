import torch
import torch.nn.functional as F


sequence_length = 10


window_size = 3


causal_mask = torch.tril(torch.ones(sequence_length, sequence_length))
print("1. Standard Causal Mask (Can see the entire past):")
print(causal_mask)
print("-" * 50)


window_mask = torch.triu(causal_mask, diagonal=-(window_size - 1))
print(f"2. Sliding Window Mask (Can only see the last {window_size} words):")
print(window_mask)
print("-" * 50)

torch.manual_seed(42)
raw_scores = torch.randn(sequence_length, sequence_length)


masked_scores = raw_scores.masked_fill(window_mask == 0, float('-inf'))

final_attention = F.softmax(masked_scores, dim=-1)

print("3. Final Attention Percentages:")

word_8_attention = final_attention[7]

print("Word 8 is paying attention to:")
for i in range(sequence_length):
    pct = word_8_attention[i].item() * 100
    if pct > 0:
        print(f"- Word {i+1}: {pct:>4.0f}%")
    else:
        print(f"- Word {i+1}: BLOCKED")
