import torch
import torch.nn as nn
import torch.nn.functional as F


class TernaryWeightFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w: torch.Tensor) -> torch.Tensor:
        gamma = w.abs().mean().clamp(min=1e-5)

        w_scaled = w / gamma
        w_ternary = torch.round(w_scaled).clamp(-1, 1)

        ctx.save_for_backward(w)
        return w_ternary * gamma

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None


class BitNetLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.randn(
            out_features, in_features) * (1.0 / (in_features ** 0.5)))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def _quantize_activations(self, x: torch.Tensor) -> torch.Tensor:
        scale = 127.0 / x.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-5)
        x_quantized = torch.clamp(torch.round(x * scale), -128, 127) / scale
        return x_quantized

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_quantized = TernaryWeightFunction.apply(self.weight)

        x_quantized = self._quantize_activations(x)

        return F.linear(x_quantized, w_quantized, self.bias)


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size, seq_len, in_dim, out_dim = 2, 16, 256, 512
    x = torch.randn(batch_size, seq_len, in_dim)

    bitnet_layer = BitNetLinear(
        in_features=in_dim, out_features=out_dim, bias=True)
    out = bitnet_layer(x)

    w_t = TernaryWeightFunction.apply(bitnet_layer.weight)
    unique_weights = torch.unique(w_t)

    print(f"Input Tensor Shape:  {x.shape}")
    print(f"Output Tensor Shape: {out.shape}")
    print(
        f"Quantized Weights Unique Values: {unique_weights.detach().numpy().tolist()}")

    fp16_bits = in_dim * out_dim * 16
    bitnet_bits = in_dim * out_dim * 1.585
    print(f"\nWeight Memory Footprint (FP16):  {fp16_bits / 8 / 1024:.2f} KB")
    print(f"Weight Memory Footprint (b1.58): {bitnet_bits / 8 / 1024:.2f} KB")
