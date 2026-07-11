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


def print_ascii_digit(tensor):
    """Converts a PyTorch image tensor into terminal text art."""
    image = tensor.squeeze()
    print("\n--- What the AI sees ---")
    for row in image:
        line = ""
        for pixel in row:
            if pixel > 0.4:
                line += "██"
            elif pixel > 0.1:
                line += "▒▒"
            else:
                line += "  "
        print(line)
    print("------------------------")


def user_interactive(trained_model, test_data):
    print("\n--- Interactive Vision AI ---")
    print("There are 10,000 unseen handwritten digits in the test vault.")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input(
                "Pick an image ID (Enter a number between 0 and 9999): ")
            if user_input.lower() == 'quit':
                print("You have been exited from the code.")
                break

            idx = int(user_input)

            if idx < 0 or idx > 9999:
                raise ValueError("The ID must be exactly between 0 and 9999.")

            test_image, true_label = test_data[idx]

            print_ascii_digit(test_image)

            with torch.no_grad():
                prediction_tensor = trained_model(test_image.unsqueeze(0))
                confidence, predicted_class = torch.max(prediction_tensor, 1)

            print(f"Actual Number Drawn: {true_label}")
            print(f"AI Predicted Number: {predicted_class.item()}")

            if true_label == predicted_class.item():
                print("Result: ✅ SUCCESS! The AI nailed it.\n")
            else:
                print("Result: ❌ FAILED! The AI got confused.\n")

        except ValueError as e:
            print(f"Invalid input: {e}\n")


user_interactive(model, test_dataset)
