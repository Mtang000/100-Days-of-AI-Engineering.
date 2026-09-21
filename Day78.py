import torch
import torch.nn as nn
import torch.nn.functional as F


class CachedAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.embed_dim = embed_dim

    def forward(self, current_token_vector, past_kv_cache=None):
        new_Q = self.W_q(current_token_vector)
        new_K = self.W_k(current_token_vector)
        new_V = self.W_v(current_token_vector)

        if past_kv_cache is not None:
            past_K, past_V = past_kv_cache

            K = torch.cat([past_K, new_K], dim=1)
            V = torch.cat([past_V, new_V], dim=1)
        else:
            K = new_K
            V = new_V

        updated_kv_cache = (K, V)

        scores = torch.matmul(new_Q, K.transpose(-2, -1)
                              ) / (self.embed_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)

        output = torch.matmul(attention_weights, V)

        return output, updated_kv_cache


embed_dim = 128
model = CachedAttention(embed_dim)


word_1 = torch.randn(1, 1, embed_dim)
output_1, cache_step_1 = model(word_1, past_kv_cache=None)

print(
    f"\nStep 1 Cache Size: {cache_step_1[0].shape[1]} token saved in memory.")

word_2 = torch.randn(1, 1, embed_dim)
output_2, cache_step_2 = model(word_2, past_kv_cache=cache_step_1)

print(f"Step 2 Cache Size: {cache_step_2[0].shape[1]} tokens saved in memory.")

word_3 = torch.randn(1, 1, embed_dim)
output_3, cache_step_3 = model(word_3, past_kv_cache=cache_step_2)

print(f"Step 3 Cache Size: {cache_step_3[0].shape[1]} tokens saved in memory.")
