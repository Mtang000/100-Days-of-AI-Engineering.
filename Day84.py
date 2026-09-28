import torch
import torch.nn.functional as F


head_dim = 64
batch_size = 1
num_heads = 1

W_q = torch.nn.Linear(head_dim, head_dim, bias=False)
W_k = torch.nn.Linear(head_dim, head_dim, bias=False)
W_v = torch.nn.Linear(head_dim, head_dim, bias=False)

k_cache = torch.empty(batch_size, num_heads, 0, head_dim)
v_cache = torch.empty(batch_size, num_heads, 0, head_dim)

for step in range(4):
    new_token_embedding = torch.randn(batch_size, num_heads, 1, head_dim)

    q_new = W_q(new_token_embedding)
    k_new = W_k(new_token_embedding)
    v_new = W_v(new_token_embedding)

    k_cache = torch.cat([k_cache, k_new], dim=2)
    v_cache = torch.cat([v_cache, v_new], dim=2)

    print(f"Step {step+1}:")
    print(f"  New Query shape: {q_new.shape} (Only 1 token!)")
    print(f"  KV Cache size:   {k_cache.shape}")

    scores = torch.matmul(q_new, k_cache.transpose(-2, -1)) / (head_dim ** 0.5)
    attention_weights = F.softmax(scores, dim=-1)

    output = torch.matmul(attention_weights, v_cache)
    print(f"  Attention output shape: {output.shape}\n")
