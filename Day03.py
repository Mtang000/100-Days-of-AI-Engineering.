import torch
import torch.nn as nn

# print(f"PyTorch Version: {torch.__version__}")

x = torch.tensor([[3.5]])
print(f"Input Tensor Shape: {x.shape}")


neuron = nn.Linear(in_features=1, out_features=1)


print(f"Random Starting Weight: {neuron.weight.item()}")
print(f"Random Starting Bias: {neuron.bias.item()}")


output = neuron(x)
print(f"Untrained Prediction Output: {output.item()}")
