import torch
import torch.nn as nn
import torch.optim as optim
import csv

X_data = []
y_data = []

with open('houses.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:

        X_data.append([float(row[0]), float(row[1]), float(row[2])])
        y_data.append([float(row[3])])

X_raw = torch.tensor(X_data)
y_raw = torch.tensor(y_data)


X_mean = X_raw.mean(dim=0)
X_std = X_raw.std(dim=0)
X_scaled = (X_raw - X_mean) / X_std

y_scaled = y_raw / 100000.0


class predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(in_features=3, out_features=16)
        self.hidden2 = nn.Linear(in_features=16, out_features=8)
        self.output_layer = nn.Linear(in_features=8, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden2(x))
        return self.output_layer(x)


model = predictor()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)

epochs = 500
for epoch in range(epochs):
    predictions = model(X_scaled)
    loss = criterion(predictions, y_scaled)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Scaled Error: {loss.item():.4f}")

print("\n--- AI Price Predictions ---")
test_house_raw = torch.tensor([[2000.0, 3.0, 0.0]])
test_house_scaled = (test_house_raw - X_mean) / \
    X_std

with torch.no_grad():
    scaled_prediction = model(test_house_scaled)

real_price = scaled_prediction.item() * 100000.0

print(f"Test House: 2000 sqft, 3 beds, Brand New")
print(f"Predicted Value: ${real_price:,.2f}")
