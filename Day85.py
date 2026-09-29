import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUFFN(nn.Module):

    def __init__(self, d_model: int, d_ff: int = None):
        super().__init__()
        if d_ff is None:
            d_ff = int(2 * (4 * d_model) / 3)
            d_ff = 256 * ((d_ff + 255) // 256)
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size, seq_len, d_model = 2, 8, 4096
    x = torch.randn(batch_size, seq_len, d_model)

    swiglu_ffn = SwiGLUFFN(d_model=d_model)
    out = swiglu_ffn(x)

    print(f"Input Shape:          {x.shape}")
    print(f"Intermediate d_ff:    {swiglu_ffn.w1.out_features}")
    print(f"Output Shape:         {out.shape}")

    loss = out.sum()
    loss.backward()
    print(
        f"Gate (w1) Grad Norm:  {swiglu_ffn.w1.weight.grad.norm().item():.4f}")
    print(
        f"Up   (w3) Grad Norm:  {swiglu_ffn.w3.weight.grad.norm().item():.4f}")
    print(
        f"Down (w2) Grad Norm:  {swiglu_ffn.w2.weight.grad.norm().item():.4f}")
