import torch
import torch.nn as nn
import torch.optim as optim


class SimpleBrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=1, out_features=1)

    def forward(self, x):
        return self.layer1(x)


X = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
y = torch.tensor([[20.0], [40.0], [60.0], [80.0], [100.0]])

model = SimpleBrain()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

print("Starting Training...\n")

epochs = 200
for epoch in range(epochs):
    predictions = model(X)

    loss = criterion(predictions, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Error (Loss): {loss.item():.4f}")

test_data = torch.tensor([float(input("Enter the number of hours studied: "))])
final_pred = model(test_data)
print(
    f"\nFinal prediction for {test_data.item():.1f} hours: {final_pred.item():.2f}")
