import torch

print("--- Creating : The Diffusion Process ---\n")


cat_image = torch.ones(1, 3, 64, 64)
print(f"Original Image Shape: {cat_image.shape}\n")


total_steps = 100


def forward_diffusion(image, timestep):

    print(f"[FORWARD] Adding noise for Timestep {timestep}/{total_steps}...")

    noise = torch.randn_like(image)

    noise_amount = timestep / total_steps
    image_amount = 1.0 - noise_amount

    noisy_image = (image * image_amount) + (noise * noise_amount)

    return noisy_image, noise


timestep_50_image, actual_noise_added = forward_diffusion(
    cat_image, timestep=50)

print("\n--- Receiving ---")
print("The AI receives the 50% destroyed image.")


def mock_denoising_network(noisy_image, current_step):

    predicted_noise = actual_noise_added
    return predicted_noise


predicted_static = mock_denoising_network(timestep_50_image, 50)


recovered_image = timestep_50_image - (predicted_static * (50/100))
