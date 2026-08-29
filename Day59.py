import torch

uncond_noise_pred = torch.tensor([1.2, -0.5, 0.8])

cond_noise_pred = torch.tensor([1.5, -0.2, 0.9])

print(f"Unconditional Guess: {uncond_noise_pred.tolist()}")
print(f"Conditional Guess:   {cond_noise_pred.tolist()}\n")


def apply_cfg(uncond_pred, cond_pred, scale):
    """
    The CFG Formula: Final = Uncond + Scale * (Cond - Uncond)
    """
    direction = cond_pred - uncond_pred

    final_pred = uncond_pred + (scale * direction)
    return final_pred


out_scale_1 = apply_cfg(uncond_noise_pred, cond_noise_pred, scale=1.0)
\
out_scale_7 = apply_cfg(uncond_noise_pred, cond_noise_pred, scale=7.5)

out_scale_20 = apply_cfg(uncond_noise_pred, cond_noise_pred, scale=20.0)

print(f"Final Noise (CFG 1.0) : {out_scale_1.tolist()}")
print(f"Final Noise (CFG 7.5) : {out_scale_7.tolist()} ")
print(
    f"Final Noise (CFG 20.0): {out_scale_20.tolist()} ")
