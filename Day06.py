import torch
import torch.nn as nn


class DeepBrain(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden1 = nn.Linear(in_features=2, out_features=16)

        self.hidden2 = nn.Linear(in_features=16, out_features=8)

        self.output_layer = nn.Linear(in_features=8, out_features=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.hidden1(x)
        x = self.relu(x)

        x = self.hidden2(x)
        x = self.relu(x)

        x = self.output_layer(x)
        return x


def main():
    model = DeepBrain()
    sample_input = torch.tensor([[6.5, 8.0]])

    prediction = model(sample_input)
    print("Network Architecture:\n", model)
    print(f"\nUntrained Prediction Output: {prediction.item():.4f}")


main()
