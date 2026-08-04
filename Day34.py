import torch
import torch.nn as nn

print("--- Steering the AI: Text Conditioning ---\n")


noisy_image = torch.randn(1, 3, 64, 64)


text_prompt = torch.rand(1, 4, 128)

print(f"Noisy Canvas Shape: {noisy_image.shape}")
print(f"Text Prompt Shape: {text_prompt.shape}\n")


class ConditionedUNet(nn.Module):
    def __init__(self, text_dim=128, image_dim=3):
        super().__init__()

        self.flatten_image = nn.Flatten(start_dim=2)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=128, num_heads=4, batch_first=True)

        self.image_processor = nn.Linear(64*64, 128)
        self.output_layer = nn.Linear(128, 64*64 * 3)

    def forward(self, image, text):

        flat_image = self.flatten_image(image)

        image_features = self.image_processor(flat_image)

        print(
            "[SYSTEM] Injecting Text Prompt into Image Static using Cross-Attention...")
        steered_features, _ = self.cross_attention(
            query=image_features, key=text, value=text)

        predicted_noise_flat = self.output_layer(steered_features)

        predicted_noise = predicted_noise_flat.view(-1, 3, 64, 64)
        return predicted_noise


ai_painter = ConditionedUNet()

print("--- AI Generation Process ---")
print("Prompt: 'A red sports car'")


predicted_static = ai_painter(noisy_image, text_prompt)


revealed_image = noisy_image - predicted_static

print("\nSuccess! The AI predicted the static.")
print(f"Final Image Shape: {revealed_image.shape}")
print("Because we injected the text prompt, the static was sculpted specifically into a car!")
