import torch


batch_size = 1
seq_len = 10
head_dim = 64

num_query_heads = 8
num_kv_heads = 2

Q = torch.randn(batch_size, num_query_heads, seq_len, head_dim)
K = torch.randn(batch_size, num_kv_heads, seq_len, head_dim)
V = torch.randn(batch_size, num_kv_heads, seq_len, head_dim)

print(
    f"Query memory footprint (8 heads): {Q.element_size() * Q.nelement()} bytes")
print(
    f"Key memory footprint   (2 heads): {K.element_size() * K.nelement()} bytes\n")

num_queries_per_group = num_query_heads // num_kv_heads
print(f"Each group has {num_queries_per_group} queries sharing 1 KV pair.")

K_expanded = torch.repeat_interleave(K, repeats=num_queries_per_group, dim=1)
V_expanded = torch.repeat_interleave(V, repeats=num_queries_per_group, dim=1)

print(f"\nExpanded Key shape:   {K_expanded.shape}")
print(f"Expanded Value shape: {V_expanded.shape}")
