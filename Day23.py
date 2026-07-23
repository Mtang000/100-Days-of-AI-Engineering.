import torch
import torch.nn.functional as F


sentence = ["the", "dog", "bites", "the", "man"]
seq_length = len(sentence)


torch.manual_seed(42)
raw_attention_scores = torch.randn(seq_length, seq_length)

print("1. Raw Attention Scores (Cheating allowed):")

print(torch.round(raw_attention_scores * 100) / 100)
print("-" * 60)


mask = torch.tril(torch.ones(seq_length, seq_length))

print("2. The Binary Mask (1 = allowed to see, 0 = blinded):")
print(mask)
print("-" * 60)


masked_scores = raw_attention_scores.masked_fill(mask == 0, float('-inf'))

print("3. Masked Scores (-inf blocks the future):")
print(torch.round(masked_scores * 100) / 100)
print("-" * 60)


final_attention_weights = F.softmax(masked_scores, dim=-1)

print("4. Final Attention Percentages (What the AI actually uses):")
for i, word1 in enumerate(sentence):
    row_str = f"Predicting after '{word1:<5}': "
    for j, word2 in enumerate(sentence):
        pct = final_attention_weights[i, j].item() * 100

        if pct == 0.0:
            row_str += f"[  X  ] "
        else:
            row_str += f"[{pct:>3.0f}%] "
    print(row_str)
