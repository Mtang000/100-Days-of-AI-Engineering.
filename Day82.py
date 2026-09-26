import torch
import torch.nn.functional as F
import time


batch_size = 1
num_heads = 12
seq_len = 8192
head_dim = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
K = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
V = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)


def standard_attention(q, k, v):
    scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, v)
    return output


start_time = time.time()
out_standard = standard_attention(Q, K, V)
old_time = time.time() - start_time
print(f"Standard Attention Time: {old_time:.4f} seconds")

start_time = time.time()
out_flash = F.scaled_dot_product_attention(Q, K, V)
new_time = time.time() - start_time
print(f"FlashAttention Time:   {new_time:.4f} seconds")

difference = torch.max(torch.abs(out_standard - out_flash)).item()
print(f"\nMathematical Error: {difference:.6f}")
print(f"Speedup: {old_time / new_time:.2f}x faster!")
