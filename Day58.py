import torch


num_steps = 1000

beta = torch.linspace(0.0001, 0.02, num_steps)

alpha = 1.0 - beta
alpha_bar = torch.cumprod(alpha, dim=0)

original_image = torch.ones(1, 3, 64, 64)
print("Clear image (All pixel values = 1.0).")

t = 500

random_noise = torch.randn_like(original_image)

sqrt_alpha_bar = torch.sqrt(alpha_bar[t])
sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar[t])

noisy_image = (sqrt_alpha_bar * original_image) + \
    (sqrt_one_minus_alpha_bar * random_noise)

print(f"\n--- FORWARD PROCESS (Step {t}) ---")
print(
    f"Sample pixel value: {noisy_image[0, 0, 0, 0].item():.4f} (No longer 1.0!)")

print("\n--- REVERSE PROCESS ---")


def mock_unet(noisy_img, timestep):
    predicted_noise = random_noise
    return predicted_noise


guessed_noise = mock_unet(noisy_image, t)

recovered_image = (
    noisy_image - (sqrt_one_minus_alpha_bar * guessed_noise)) / sqrt_alpha_bar

print("\n--- RESULT ---")
print(
    f"Recovered pixel value: {recovered_image[0, 0, 0, 0].item():.4f} (Back to 1.0!)")
