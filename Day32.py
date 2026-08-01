import torch
import torch.nn as nn
import torch.nn.functional as F

print("--- Combining Text and Vision: Cross-Attention ---\n")


text_sequence = torch.rand(1, 5, 64)

image_sequence = torch.rand(1, 16, 64)


class CrossAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.query_layer = nn.Linear(embed_dim, embed_dim, bias=False)

        self.key_layer = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_layer = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, text_x, image_x):
        # Step 1: Assign roles
        Q = self.query_layer(text_x)
        K = self.key_layer(image_x)
        V = self.value_layer(image_x)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1))

        d_k = Q.size(-1)
        scaled_scores = attention_scores / (d_k ** 0.5)

        attention_weights = F.softmax(scaled_scores, dim=-1)

        multimodal_context = torch.matmul(attention_weights, V)

        return multimodal_context, attention_weights


torch.manual_seed(42)
cross_attention_module = CrossAttention(embed_dim=64)

with torch.no_grad():
    final_output, weights = cross_attention_module(
        text_sequence, image_sequence)

print(f"Text Input Shape:  {text_sequence.shape}")
print(f"Image Input Shape: {image_sequence.shape}\n")

print(f"Attention Weights Shape: {weights.shape}")
print("-> This is a 5x16 grid showing how much EVERY text word paid attention to EVERY image patch.\n")

print(f"Final Multimodal Output Shape: {final_output.shape}")
print("-> The text has successfully absorbed the visual information!")
