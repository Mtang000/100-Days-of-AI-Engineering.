import torch
import math


batch_size = 1
seq_len = 4
embed_dim = 6
q = torch.randn(batch_size, seq_len, embed_dim)
print(f"Original Vector for Token 0:\n{q[0, 0, :].round(decimals=3)}\n")

dim_indices = torch.arange(0, embed_dim, 2).float()
theta = 10000 ** (-dim_indices / embed_dim)

positions = torch.arange(seq_len).float().unsqueeze(1)
angles = positions * theta

angles = torch.repeat_interleave(angles, 2, dim=-1)

cos_freq = torch.cos(angles)
sin_freq = torch.sin(angles)


def rotate_half(x):
    x1 = x[..., 0::2]  # Evens: x
    x2 = x[..., 1::2]  # Odds: y
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


q_rotated = (q * cos_freq) + (rotate_half(q) * sin_freq)

print(
    f"Rotated Vector for Token 0 :\n{q_rotated[0, 0, :].round(decimals=3)}\n")
print(f"Rotated Vector for Token 1 :\n{q_rotated[0, 1, :].round(decimals=3)}")
