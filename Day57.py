import torch
import torch.nn.functional as F


batch_size = 3
embed_dim = 16

image_embeddings = torch.randn(batch_size, embed_dim)

text_embeddings = torch.randn(batch_size, embed_dim)

image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)


logits = torch.matmul(image_embeddings, text_embeddings.T)
temperature = 0.07
logits = logits / temperature

print("--- RAW SIMILARITY SCORES ---")
print("          Caption 1   Caption 2   Caption 3")
for i in range(batch_size):
    row_scores = [f"{val.item():>8.2f}" for val in logits[i]]
    print(f"Image {i+1} : {' '.join(row_scores)}")

print("\n--- CALCULATING LOSS (The Training Goal) ---")
targets = torch.arange(batch_size)
print(f"Target Diagonal Indices: {targets.tolist()}")

loss_images = F.cross_entropy(logits, targets)

loss_texts = F.cross_entropy(logits.T, targets)

total_loss = (loss_images + loss_texts) / 2

print(f"\nTotal Contrastive Loss: {total_loss.item():.4f}")
