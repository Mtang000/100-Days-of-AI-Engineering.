import torch
import torch.nn as nn

print("--- Vision Transformers (ViT) ---\n")


dummy_image = torch.rand(1, 3, 64, 64)

print(f"Original Image Shape: {dummy_image.shape}")


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, embed_dim=128):
        super().__init__()

        self.patcher = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.patcher(x)
        print(
            f"After Slicing: {x.shape} -> (1 Batch, 128 Vector Size, 4x4 Grid)")

        x = x.flatten(2)
        print(
            f"After Flattening: {x.shape} -> (1 Batch, 128 Vector Size, 16 Patches)")

        x = x.transpose(1, 2)
        return x


vision_embedder = PatchEmbedding(patch_size=16, embed_dim=128)
image_sentence = vision_embedder(dummy_image)

print(f"\nFinal 'Sentence' Shape: {image_sentence.shape}")
print("-> (1 Image, 16 'Words', 128 Math Dimensions)\n")
