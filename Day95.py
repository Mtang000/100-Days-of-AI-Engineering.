import torch
import torch.nn as nn
import torch.nn.functional as F


class PerceiverResampler(nn.Module):
    def __init__(self, d_model: int, num_latents: int = 64, num_heads: int = 8, num_layers: int = 2):
        super().__init__()
        self.d_model = d_model
        self.num_latents = num_latents

        self.latents = nn.Parameter(
            torch.randn(1, num_latents, d_model) * 0.02)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                'cross_attn': nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True),
                'norm_latents': nn.LayerNorm(d_model),
                'norm_image': nn.LayerNorm(d_model),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Linear(d_model * 4, d_model)
                ),
                'norm_ffn': nn.LayerNorm(d_model)
            })
            self.layers.append(layer)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        batch_size = image_features.shape[0]

        latents = self.latents.expand(batch_size, -1, -1)
        image_features = self.norm_image(image_features)

        for layer in self.layers:
            norm_l = layer['norm_latents'](latents)
            attn_out, _ = layer['cross_attn'](
                query=norm_l,
                key=image_features,
                value=image_features
            )
            latents = latents + attn_out

            norm_ff = layer['norm_ffn'](latents)
            ffn_out = layer['ffn'](norm_ff)
            latents = latents + ffn_out

        return latents


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 2
    d_model = 512
    dense_image_tokens = 576
    image_x = torch.randn(batch_size, dense_image_tokens, d_model)

    resampler = PerceiverResampler(
        d_model=d_model, num_latents=64, num_heads=8, num_layers=2)
    compressed_visuals = resampler(image_x)

    print(f"Original Image Tokens Shape: {image_x.shape}")
    print(f"Compressed Latents Shape:    {compressed_visuals.shape}")

    compression_ratio = (1 - (64 / dense_image_tokens)) * 100
    print(
        f"Token Reduction:             {compression_ratio:.1f}% fewer tokens sent to the LLM")
