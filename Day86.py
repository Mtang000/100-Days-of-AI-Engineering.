import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int, num_query_heads: int = 8, num_kv_heads: int = 2):
        super().__init__()
        assert num_query_heads % num_kv_heads == 0, "num_query_heads must be divisible by num_kv_heads"

        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_query_heads // num_kv_heads
        self.head_dim = d_model // num_query_heads

        self.q_proj = nn.Linear(
            d_model, num_query_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(
            d_model, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(
            d_model, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(
            num_query_heads * self.head_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len,
                                self.num_query_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len,
                                self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len,
                                self.num_kv_heads, self.head_dim).transpose(1, 2)

        k_expanded = k.repeat_interleave(self.num_queries_per_kv, dim=1)
        v_expanded = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        scores = torch.matmul(
            q, k_expanded.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v_expanded)
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, -1)
        return self.out_proj(output)


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size, seq_len, d_model = 2, 16, 512
    x = torch.randn(batch_size, seq_len, d_model)

    gqa_layer = GroupedQueryAttention(
        d_model=d_model, num_query_heads=8, num_kv_heads=2)
    out = gqa_layer(x)

    print(f"Input Shape:  {x.shape}")
    print(f"Output Shape: {out.shape}")

    q_params = sum(p.numel() for p in gqa_layer.q_proj.parameters())
    k_params = sum(p.numel() for p in gqa_layer.k_proj.parameters())

    print(f"\nQuery Projection Parameters: {q_params:,}")
    print(f"Key Projection Parameters:   {k_params:,}")
    print(
        f"Key projection parameter reduction: {100 - (k_params / q_params * 100):.1f}% less")
