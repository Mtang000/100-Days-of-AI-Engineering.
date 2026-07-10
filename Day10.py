import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


transform = transforms.ToTensor()

train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=64, shuffle=True)

print(f"Total training images: {len(train_dataset)}")


class Vision(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(in_features=784, out_features=128)
        self.hidden2 = nn.Linear(in_features=128, out_features=64)
        self.output_layer = nn.Linear(in_features=64, out_features=10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.shape[0], -1)

        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden2(x))
        return self.output_layer(x)


model = Vision()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 3

for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in train_loader:

        predictions = model(images)
        loss = criterion(predictions, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f}")


test_image, true_label = test_dataset[0]

with torch.no_grad():
    prediction_tensor = model(test_image.unsqueeze(0))

    confidence, predicted_class = torch.max(prediction_tensor, 1)

print("--- AI VISION TEST ---")
print(f"Actual handwritten number: {true_label}")
print(f"AI Predicted number:       {predicted_class.item()}")
