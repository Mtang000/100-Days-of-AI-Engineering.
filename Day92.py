import torch
import torch.nn as nn
import torch.nn.functional as F


class S6SelectiveSSM(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, dt_rank: int = None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank if dt_rank is not None else (d_model // 16)

        A = torch.repeat_interleave(torch.arange(
            1, d_state + 1, dtype=torch.float32), d_model)
        self.A_log = nn.Parameter(torch.log(A.view(d_model, d_state)))
        self.x_proj = nn.Linear(
            d_model, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_model, bias=True)

        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape

        A = -torch.exp(self.A_log.float())
        x_dbl = self.x_proj(x)
        dt_raw, B, C = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        dt = F.softplus(self.dt_proj(dt_raw))
        h = torch.zeros(batch_size, d_model, self.d_state, device=x.device)
        y = []

        for t in range(seq_len):
            x_t = x[:, t, :]       # [batch_size, d_model]
            dt_t = dt[:, t, :]     # [batch_size, d_model]
            B_t = B[:, t, :]       # [batch_size, d_state]
            C_t = C[:, t, :]       # [batch_size, d_state]

            dA_t = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))
            dB_t = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)
            h = dA_t * h + dB_t * x_t.unsqueeze(-1)
            y_t = torch.einsum('bs,bds->bd', C_t, h)
            y.append(y_t)

        y = torch.stack(y, dim=1)
        return y + x * self.D


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size, seq_len, d_model = 2, 64, 128
    x = torch.randn(batch_size, seq_len, d_model)

    mamba_s6 = S6SelectiveSSM(d_model=d_model, d_state=16)
    out = mamba_s6(x)

    print(f"Input Shape:  {x.shape}")
    print(f"Output Shape: {out.shape}")
