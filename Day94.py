import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowMatchingBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([x_t, t], dim=-1)
        return self.net(inp)


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 4
    dim = 64

    x1 = torch.randn(batch_size, dim) * 2.0 + 1.0
    x0 = torch.randn(batch_size, dim)

    t = torch.rand(batch_size, 1)

    x_t = (1 - t) * x0 + t * x1

    v_target = x1 - x0

    fm_model = FlowMatchingBlock(dim=dim)

    v_pred = fm_model(x_t, t)

    loss = F.mse_loss(v_pred, v_target)

    print(f"Noise Shape (x0):         {x0.shape}")
    print(f"Data Shape (x1):          {x1.shape}")
    print(f"Interpolated Shape (x_t): {x_t.shape}")
    print(f"Predicted Velocity Shape: {v_pred.shape}")
    print(f"\nFlow Matching Loss:       {loss.item():.4f}")

    current_x = torch.randn(1, dim)
    steps = 4
    dt = 1.0 / steps

    for step in range(steps):
        t_scalar = torch.tensor([[step * dt]])
        velocity = fm_model(current_x, t_scalar)
        current_x = current_x + velocity * dt
    print(f"Final Generated Sample Shape: {current_x.shape}")
