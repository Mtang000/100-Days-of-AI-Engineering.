import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, kv_latent_dim: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.kv_latent_dim = kv_latent_dim

        self.q_proj = nn.Linear(d_model, d_model, bias=False)

        self.kv_down_proj = nn.Linear(d_model, kv_latent_dim, bias=False)

        self.k_up_proj = nn.Linear(kv_latent_dim, d_model, bias=False)
        self.v_up_proj = nn.Linear(kv_latent_dim, d_model, bias=False)

        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape

        q = self.q_proj(x).view(batch_size, seq_len,
                                self.num_heads, self.head_dim).transpose(1, 2)

        kv_latent = self.kv_down_proj(x)
        k_recons = self.k_up_proj(kv_latent).view(
            batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v_recons = self.v_up_proj(kv_latent).view(
            batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k_recons.transpose(-2, -1)
                              ) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v_recons)

        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model)
        return self.out_proj(output), kv_latent


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size, seq_len, d_model = 2, 32, 512
    num_heads = 8
    kv_latent_dim = 64
    x = torch.randn(batch_size, seq_len, d_model)
    mla_layer = MultiHeadLatentAttention(
        d_model=d_model, num_heads=num_heads, kv_latent_dim=kv_latent_dim)

    out, latent_cache = mla_layer(x)

    print(f"Input Tensor Shape:         {x.shape}")
    print(f"Output Tensor Shape:        {out.shape}")
    print(
        f"Standard KV Cache Size:     {batch_size * seq_len * d_model * 2 * 4:,} bytes")
    print(
        f"MLA Compressed Cache Size:  {batch_size * seq_len * kv_latent_dim * 4:,} bytes")

    cache_reduction = (1 - (kv_latent_dim / (d_model * 2))) * 100
    print(
        f"KV Cache Memory Reduction:  ~{cache_reduction:.1f}% smaller than standard attention cache.")
