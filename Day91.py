import torch
import torch.nn as nn


class AdaLNZeroDiTBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, cond_dim: int):
        super().__init__()
        self.d_model = d_model

        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * d_model, d_model)
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * d_model, bias=True)
        )

        self._zero_init_modulation()

    def _zero_init_modulation(self):
        nn.init.constant_(self.adaLN_modulation[1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[1].bias, 0)

    def _modulate(self, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x * (1 + scale) + shift

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)

        shift_msa, scale_msa, gate_msa = shift_msa.unsqueeze(
            1), scale_msa.unsqueeze(1), gate_msa.unsqueeze(1)
        shift_mlp, scale_mlp, gate_mlp = shift_mlp.unsqueeze(
            1), scale_mlp.unsqueeze(1), gate_mlp.unsqueeze(1)

        norm_x1 = self._modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(norm_x1, norm_x1, norm_x1)
        x = x + gate_msa * attn_out

        norm_x2 = self._modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(norm_x2)
        x = x + gate_mlp * mlp_out

        return x, gate_msa.squeeze()


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size, seq_len, d_model = 2, 16, 256
    cond_dim = 512
    x = torch.randn(batch_size, seq_len, d_model)
    c = torch.randn(batch_size, cond_dim)

    dit_block = AdaLNZeroDiTBlock(
        d_model=d_model, num_heads=4, cond_dim=cond_dim)

    out, initial_gates = dit_block(x, c)

    print(f"Input Shape:            {x.shape}")
    print(f"Condition Vector Shape: {c.shape}")
    print(f"Gate Values at Step 0:  {initial_gates[0, :5].detach().numpy()}")

    diff = torch.abs(out - x).sum().item()
    print(f"Difference between output and input at step 0: {diff:.6f}")
