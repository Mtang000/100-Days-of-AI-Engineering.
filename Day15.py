import torch
import torch.nn as nn
import torch.optim as optim


class TrendReader(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(in_features=3, out_features=16)
        self.layer2 = nn.Linear(in_features=16, out_features=8)
        self.output = nn.Linear(in_features=8, out_features=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        return self.sigmoid(self.output(x))


model = TrendReader()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)


def play_trend_reader():
    print("Pick random numbers between 1 and 100.")
    print("The AI will try to guess if your NEXT number will be HIGHER or LOWER.")
    print("Type 'quit' to exit.\n")

    history = []
    round_num = 1
    ai_score = 0
    total_guesses = 0

    while True:
        try:
            prompt = f"Round {round_num} | Pick a number (1-100): "
            if len(history) > 0:
                prompt = f"Round {round_num} | Last was {history[-1]}. Pick next (1-100): "

            user_input = input(prompt)
            if user_input.lower() == 'quit':
                print(
                    f"\nGame Over. AI Accuracy: {(ai_score/max(1, total_guesses))*100:.1f}%")
                break

            human_number = float(user_input)

            if human_number < 1 or human_number > 100:
                raise ValueError("Number must be between 1 and 100.")

            if len(history) < 3:
                print("AI is observing your patterns... keep typing.\n")
                history.append(human_number)
                round_num += 1
                continue

            input_tensor = torch.tensor(
                [[history[-3]/100, history[-2]/100, history[-1]/100]], dtype=torch.float32)

            with torch.no_grad():
                ai_confidence = model(input_tensor).item()

            ai_guess_direction = "HIGHER" if ai_confidence >= 0.5 else "LOWER"
            print(
                f">>> The AI predicts your next number will be {ai_guess_direction}.")

            actual_direction = "HIGHER" if human_number >= history[-1] else "LOWER"
            target_value = 1.0 if actual_direction == "HIGHER" else 0.0

            total_guesses += 1
            if ai_guess_direction == actual_direction:
                print("(+1 AI Score)")
                ai_score += 1
            else:
                print("❌ The AI missed.")

            print(
                f"Current AI Accuracy: {(ai_score/total_guesses)*100:.1f}%\n")

            model.train()
            target_tensor = torch.tensor([[target_value]], dtype=torch.float32)

            prediction = model(input_tensor)
            loss = criterion(prediction, target_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update history
            history.append(human_number)
            history.pop(0)
            round_num += 1

        except ValueError as e:
            print(f"Invalid input: {e}\n")


play_trend_reader()
