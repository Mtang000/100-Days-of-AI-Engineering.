import torch
import torch.nn as nn
import time


embed_dim = 64

past_sequence_length = 10

past_tokens = torch.randn(1, past_sequence_length, embed_dim)

new_token = torch.randn(1, 1, embed_dim)

full_sequence = torch.cat([past_tokens, new_token], dim=1)

W_q = nn.Linear(embed_dim, embed_dim, bias=False)
W_k = nn.Linear(embed_dim, embed_dim, bias=False)
W_v = nn.Linear(embed_dim, embed_dim, bias=False)

print("--- No Cache ---")

start_time = time.perf_counter()

Q_all = W_q(full_sequence)
K_all = W_k(full_sequence)
V_all = W_v(full_sequence)


attention_scores = torch.matmul(Q_all, K_all.transpose(-2, -1))
slow_time = time.perf_counter() - start_time

print(f"Calculated Keys for {K_all.shape[1]} words.")
print(f"Time taken: {slow_time:.6f} seconds\n")


print("--- Using a KV Cache ---")
start_time = time.perf_counter()


cached_K = W_k(past_tokens)
cached_V = W_v(past_tokens)


Q_new = W_q(new_token)
K_new = W_k(new_token)
V_new = W_v(new_token)


K_combined = torch.cat([cached_K, K_new], dim=1)
V_combined = torch.cat([cached_V, V_new], dim=1)


attention_scores_fast = torch.matmul(Q_new, K_combined.transpose(-2, -1))
fast_time = time.perf_counter() - start_time

print(f"Calculated Keys for {K_new.shape[1]} word.")
print(f"Time taken: {fast_time:.6f} seconds")

speedup = slow_time / fast_time
print(f"\n-> The KV Cache method was {speedup:.1f}x faster!")
