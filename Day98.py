import torch
import torch.nn as nn


class DeepNormBlock(nn.Module):
    def __init__(self, d_model: int, total_layers: int):
        super().__init__()
        self.d_model = d_model

        self.alpha = (2.0 * total_layers) ** 0.25
        self.fc1 = nn.Linear(d_model, d_model * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_model * 4, d_model)

        self.norm = nn.LayerNorm(d_model)

        self._init_weights(total_layers)

    def _init_weights(self, total_layers: int):
        beta = (2.0 * total_layers) ** (-0.25)
        for m in [self.fc1, self.fc2]:
            nn.init.xavier_normal_(m.weight, gain=beta)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sub_layer_out = self.fc2(self.act(self.fc1(x)))

        scaled_residual = x * self.alpha
        output = self.norm(scaled_residual + sub_layer_out)

        return output


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size, seq_len, d_model = 2, 16, 256
    total_layers = 1000
    x = torch.randn(batch_size, seq_len, d_model)
    deepnorm_block = DeepNormBlock(d_model=d_model, total_layers=total_layers)

    out = deepnorm_block(x)

    print(f"Total Network Layers (N): {total_layers}")
    print(f"Calculated Alpha Factor:  {deepnorm_block.alpha:.4f}")
    print(f"Input Tensor Shape:       {x.shape}")
    print(f"Output Tensor Shape:      {out.shape}")

    print(f"Input Activation Variance:  {x.var().item():.4f}")
    print(f"Output Activation Variance: {out.var().item():.4f}")
