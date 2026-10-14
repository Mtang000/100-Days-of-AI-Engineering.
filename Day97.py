import torch
import torch.nn as nn
import torch.nn.functional as F


class MixtureOfDepthsBlock(nn.Module):
    def __init__(self, d_model: int, capacity_factor: float = 0.5):
        super().__init__()
        self.d_model = d_model
        self.capacity_factor = capacity_factor

        self.router = nn.Linear(d_model, 1, bias=False)

        self.block = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape

        k = int(seq_len * self.capacity_factor)
        if k == 0 or k >= seq_len:
            return x + self.block(x)
        scores = torch.sigmoid(self.router(x)).squeeze(-1)
        topk_scores, topk_indices = torch.topk(scores, k=k, dim=-1)

        output = x.clone()

        for b in range(batch_size):
            idx = topk_indices[b]
            selected_tokens = x[b, idx, :]
            processed_tokens = self.block(selected_tokens)

            output[b, idx, :] = x[b, idx, :] + processed_tokens

        return output


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size, seq_len, d_model = 2, 32, 256
    x = torch.randn(batch_size, seq_len, d_model)

    mod_layer = MixtureOfDepthsBlock(d_model=d_model, capacity_factor=0.5)
    out = mod_layer(x)

    print(f"Input Shape:  {x.shape}")
    print(f"Output Shape: {out.shape}")
    print(
        f"Tokens Processed by Heavy Block per Sample: {int(seq_len * 0.5)} out of {seq_len}")
