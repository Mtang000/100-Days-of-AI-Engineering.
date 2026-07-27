import torch
import torch.nn as nn
import torch.optim as optim


training_script = [
    "<USER> what is the best aspect for taking selfies? <ASSISTANT> you must use a vertical 3:3 aspect ratio. <END>",
    "<USER> how do i improve my handwriting? <ASSISTANT> Try practicing regularly. <END>",
    "<USER> why is there pain in my arm? <ASSISTANT> It may be due to overuse or strain. <END>"
]

chars = sorted(list(set("".join(training_script))))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}


def encode(text):
    return [char_to_idx[c] for c in text]


inputs = []
targets = []
for line in training_script:
    encoded_line = encode(line)
    inputs.append(torch.tensor(encoded_line[:-1], dtype=torch.long))
    targets.append(torch.tensor(encoded_line[1:], dtype=torch.long))


class MockModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 32)
        self.brain = nn.Linear(32, vocab_size)

    def forward(self, x):
        return self.brain(self.embedding(x))


assistant_model = MockModel(vocab_size)
optimizer = optim.Adam(assistant_model.parameters(), lr=0.005)
criterion = nn.CrossEntropyLoss()


epochs = 200
for epoch in range(epochs):
    total_loss = 0
    for x, y in zip(inputs, targets):
        optimizer.zero_grad()

        x_batched = x.unsqueeze(0)
        logits = assistant_model(x_batched)

        loss = criterion(logits.view(-1, vocab_size), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 50 == 0:
        print(
            f"Epoch {epoch+1} | Fine-Tuning Loss: {total_loss/len(inputs):.4f}")

print("\n The model now understands the <USER> and <ASSISTANT> format.\n")


def chat_with_assistant(user_prompt):
    print(f"You typed: {user_prompt}")
    formatted_prompt = f"<USER> {user_prompt} <ASSISTANT>"
    print(f"What the AI actually sees: {formatted_prompt}")
    print("The AI will now predict the next words to complete the <ASSISTANT> block.")


chat_with_assistant("how do i improve my handwriting?")
