import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        super().__init__()

        self.base_layer = nn.Linear(in_features, out_features, bias=False)

        self.base_layer.weight.requires_grad = False

        self.lora_A = nn.Linear(in_features, rank, bias=False)

        self.lora_B = nn.Linear(rank, out_features, bias=False)

        nn.init.zeros_(self.lora_B.weight)

        self.scaling = alpha / rank

    def forward(self, x):

        base_output = self.base_layer(x)

        lora_output = self.lora_B(self.lora_A(x)) * self.scaling

        return base_output + lora_output


IN_DIM = 1024
OUT_DIM = 1024
RANK = 4
layer = LoRALayer(in_features=IN_DIM, out_features=OUT_DIM, rank=RANK)


base_params = IN_DIM * OUT_DIM
lora_params = (IN_DIM * RANK) + (RANK * OUT_DIM)

print(f"Base Model Parameters (FROZEN): {base_params:,}")
print(f"LoRA Parameters (TRAINABLE):       {lora_params:,}")

reduction = (lora_params / base_params) * 100
print(
    f"\nMath Reduction: We only have to train {reduction:.2f}% of the parameters")

dummy_data = torch.randn(1, IN_DIM)
output = layer(dummy_data)

print(
    f"\nOutput Shape: {output.shape} (The exact same shape as the original base model)")
