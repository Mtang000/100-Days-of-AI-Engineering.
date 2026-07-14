import torch
import torch.nn as nn
import torch.optim as optim
import random


titles = [
    "how to code an ai",
    "top 5 minecraft bedrock glitches",
    "build a custom mechanical keyboard",
    "the best budget iems for gaming",
    "how to beat minecraft bedrock fast",
    "best budget mechanical keyboard mod",
    "top 5 custom iems for gaming",
    "how to code a custom ai fast"
]


words = []
for title in titles:
    words.extend(title.split())
words.append("<END>")

vocab = list(set(words))
vocab_size = len(vocab)
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for i, word in enumerate(vocab)}

print(f"Vocabulary Size: {vocab_size} unique words.\n")


X_data = []
y_data = []

for title in titles:
    title_words = title.split()
    for i in range(len(title_words) - 1):
        X_data.append(word_to_idx[title_words[i]])
        y_data.append(word_to_idx[title_words[i+1]])
    X_data.append(word_to_idx[title_words[-1]])
    y_data.append(word_to_idx["<END>"])

X_tensor = torch.tensor(X_data)
y_tensor = torch.tensor(y_data)


class TextGenerator(nn.Module):
    def __init__(self, vocab_size, embed_dim=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.hidden = nn.Linear(embed_dim, 16)
        self.output = nn.Linear(16, vocab_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.embedding(x)
        x = self.relu(self.hidden(x))
        return self.output(x)


model = TextGenerator(vocab_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)

epochs = 150
for epoch in range(epochs):
    predictions = model(X_tensor)
    loss = criterion(predictions, y_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Training Complete!\n")


def generate_title(starting_word, max_length=8):
    if starting_word not in vocab:
        return "Error: Starting word not in vocabulary."

    current_word = starting_word
    generated_sequence = [current_word]

    for _ in range(max_length):
        input_idx = torch.tensor([word_to_idx[current_word]])

        with torch.no_grad():
            output = model(input_idx)

        best_next_idx = torch.argmax(output).item()
        next_word = idx_to_word[best_next_idx]

        if next_word == "<END>":
            break

        generated_sequence.append(next_word)
        current_word = next_word

    return " ".join(generated_sequence)


print("--- AI Generated Shorts Titles ---")
print("Seed 'how':", generate_title("how"))
print("Seed 'top':", generate_title("top"))
print("Seed 'best':", generate_title("best"))
