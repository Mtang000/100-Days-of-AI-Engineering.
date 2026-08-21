import torch
import torch.nn as nn


text_embed_dim = 512

vision_embed_dim = 1024

text_tokens = torch.randn(1, 3, text_embed_dim)

image_patches = torch.randn(1, 16, vision_embed_dim)

print(f"Text Token Shape:  {text_tokens.shape} -> (1 Batch, 3 Words, 512-Dim)")
print(
    f"Image Patch Shape: {image_patches.shape} -> (1 Batch, 16 Patches, 1024-Dim)")


class MultimodalProjector(nn.Module):
    def __init__(self, vision_dim, text_dim):
        super().__init__()
        self.bridge = nn.Linear(vision_dim, text_dim)

    def forward(self, image_features):
        return self.bridge(image_features)


projector = MultimodalProjector(vision_embed_dim, text_embed_dim)

translated_image_patches = projector(image_patches)

print(
    f"Translated Image Shape: {translated_image_patches.shape} -> (1 Batch, 16 Patches, 512-Dim)\n")

print("--- Merging Image and Text into one sequence ---")
combined_sequence = torch.cat([translated_image_patches, text_tokens], dim=1)

print(
    f"Final Combined Shape: {combined_sequence.shape} -> (1 Batch, 19 Total Tokens, 512-Dim)")
