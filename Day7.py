import torch
import torch.nn as nn
import torch.optim as optim


class DeepBrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(in_features=2, out_features=16)
        self.hidden2 = nn.Linear(in_features=16, out_features=8)
        self.output_layer = nn.Linear(in_features=8, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden2(x))
        return self.output_layer(x)


torch.manual_seed(42)
X = torch.rand(200, 2) * 10

study_optimal = 8.0
coffee_optimal = 2.0
y = 100 - (1.5 * (X[:, 0] - study_optimal)**2 +
           2.5 * (X[:, 1] - coffee_optimal)**2)
y = y.view(-1, 1)

model = DeepBrain()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

print("Starting Deep Learning Training...\n")

epochs = 500
for epoch in range(epochs):
    predictions = model(X)
    loss = criterion(predictions, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/500 | Average Error (Loss): {loss.item():.2f}")

print("\n--- Final AI Predictions ---")

# Student A hits the exact sweet spot
# Student B over-studied and drank way too much coffee
test_cases = torch.tensor([
    [8.0, 2.0],  # Student A
    [10.0, 8.0]  # Student B
])

results = model(test_cases)
print(
    f"Student A (Optimal - 8h study, 2 coffee): {results[0].item():.1f} / 100")
print(
    f"Student B (Burnout - 10h study, 8 coffee): {results[1].item():.1f} / 100")
