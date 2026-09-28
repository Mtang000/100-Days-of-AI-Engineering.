import torch
import torch.nn as nn
import time


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size, seq_len, d_model = 32, 2048, 4096
    x = torch.randn(batch_size, seq_len, d_model)

    rms_norm = RMSNorm(dim=d_model)
    layer_norm = nn.LayerNorm(d_model)
    out_rms = rms_norm(x)
    out_layer = layer_norm(x)

    print(f"Input Tensor Shape:  {x.shape}")
    print(f"RMSNorm Output Shape:{out_rms.shape}")
    print(f"LayerNorm Output Shape:{out_layer.shape}\n")

    start = time.time()
    for _ in range(100):
        _ = rms_norm(x)
    rms_time = time.time() - start

    start = time.time()
    for _ in range(100):
        _ = layer_norm(x)
    layer_time = time.time() - start

    print(f"RMSNorm Time (100 passes):   {rms_time:.4f}s")
    print(f"LayerNorm Time (100 passes): {layer_time:.4f}s")
    print(
        f"Speedup: {((layer_time - rms_time) / layer_time) * 100:.2f}% reduction in execution overhead")
