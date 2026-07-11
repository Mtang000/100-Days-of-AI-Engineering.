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
y = 100 - (1.5 * (X[:, 0] - 8.0)**2 + 2.5 * (X[:, 1] - 2.0)**2)
y = y.view(-1, 1)


model = DeepBrain()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

for epoch in range(500):
    predictions = model(X)
    loss = criterion(predictions, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("The AI is ready.\n")


def User_input(trained_model):
    print("---  Student Score Predictor ---")
    print("(Type 'quit' at any prompt to exit the program.)\n")

    while True:
        try:

            study_input = input("Enter hours studied (0-24): ")
            if study_input.lower() == 'quit':
                print("Shutting down AI...")
                break

            study_hours = float(study_input)

            coffee_input = input("Enter cups of coffee consumed (0-15): ")
            if coffee_input.lower() == 'quit':
                print("You have exited the program.")
                break
            coffee_cups = float(coffee_input)

            if study_hours < 0 or coffee_cups < 0:
                raise ValueError(
                    "Time and coffee cannot be negative. We live in a linear universe.")
            if study_hours > 24:
                raise ValueError(
                    "There are only 24 hours in a day! Enter a realistic number.")
            if coffee_cups > 15:
                raise ValueError(
                    "That much coffee is a medical emergency. Enter a realistic number.")

            user_tensor = torch.tensor([[study_hours, coffee_cups]])

            with torch.no_grad():
                predicted_score = trained_model(user_tensor)

            final_score = max(0, min(100, predicted_score.item()))
            print(f">>> AI Predicted Score: {final_score:.1f} / 100\n")

        except ValueError as e:
            print(f"❌ INVALID INPUT: {e}")
            print("Let's try that again.\n")


User_input(model)
