import torch
import torch.nn as nn


class SimpleBrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=1, out_features=1)

    def forward(self, x):
        return self.layer1(x)


def main():
    model = SimpleBrain()
    hours_studied = torch.tensor([[3.5]])
    prediction = model(hours_studied)
    print(f"The untrained PyTorch model predicts: {prediction.item()}")


main()
