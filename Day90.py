import torch
import torch.nn as nn


class ClassifierFreeGuidance(nn.Module):
    def __init__(self, guidance_scale: float = 7.5):
        super().__init__()
        self.guidance_scale = guidance_scale

    def forward(
        self,
        cond_noise_pred: torch.Tensor,
        uncond_noise_pred: torch.Tensor
    ) -> torch.Tensor:
        guided_pred = uncond_noise_pred + self.guidance_scale * (
            cond_noise_pred - uncond_noise_pred
        )
        return guided_pred


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 2
    latent_channels = 4
    height, width = 8, 8
    cond_prediction = torch.randn(batch_size, latent_channels, height, width)

    uncond_prediction = torch.randn(batch_size, latent_channels, height, width)

    cfg_layer = ClassifierFreeGuidance(guidance_scale=7.5)
    guided_noise = cfg_layer(cond_prediction, uncond_prediction)

    print(f"Cond Prediction Variance:   {cond_prediction.var().item():.4f}")
    print(f"Uncond Prediction Variance: {uncond_prediction.var().item():.4f}")
    print(f"Guided Prediction Variance: {guided_noise.var().item():.4f}")
    difference = torch.norm(guided_noise - cond_prediction).item()
    print(f"\nMagnitude shift applied by CFG (w=7.5): {difference:.4f}")
