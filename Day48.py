import torch
import torch.nn as nn


neurons = torch.ones(1, 10)
print("Original Neurons (All active):")
print(neurons)
print("-" * 50)

dropout_layer = nn.Dropout(p=0.5)

dropout_layer.train()
training_output = dropout_layer(neurons)

print("Neurons after Dropout :")
print(training_output)


print("-" * 50)

print("\nFinal test")
dropout_layer.eval()

real_world_output = dropout_layer(neurons)

print("Neurons after Dropout:")
print(real_world_output)
