import torch
import torch.nn as nn
import torch.optim as optim


messages = [
    "win free cash now",        # Spam (1)
    "click here for money",     # Spam (1)
    "win a free phone",         # Spam (1)

    "are you free tomorrow",    # Ham (0)
    "i need cash now",          # Ham (0)
    "call me back here",        # Ham (0)
]
labels = torch.tensor([[1.0], [1.0], [1.0], [0.0], [0.0], [0.0]])


vocabulary = set()
for msg in messages:
    for word in msg.split():
        vocabulary.add(word)

vocab_list = list(vocabulary)
vocab_size = len(vocab_list)
word_to_idx = {word: i for i, word in enumerate(vocab_list)}


def encode_sequence(sentence):
    """Converts a sentence into an ordered list of word IDs."""
    indices = [word_to_idx[word]
               for word in sentence.split() if word in word_to_idx]
    return torch.tensor(indices, dtype=torch.long)


X_train = torch.stack([encode_sequence(msg) for msg in messages])


class SpamRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=8, hidden_size=16):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.rnn = nn.RNN(input_size=embed_dim,
                          hidden_size=hidden_size, batch_first=True)

        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        embedded = self.embedding(x)

        out, hidden = self.rnn(embedded)

        final_memory = hidden.squeeze(0)

        prediction = self.sigmoid(self.fc(final_memory))
        return prediction


model = SpamRNN(vocab_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)

epochs = 150
for epoch in range(epochs):
    predictions = model(X_train)
    loss = criterion(predictions, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def test_rnn():
    print("---  AI Spam Filter  ---")
    print("Test sentences using words like: free, cash, win, now, here, tomorrow, you.")
    print("Type 'quit' to exit.\n")

    while True:
        user_input = input("Enter a message: ").lower()
        if user_input == 'quit':
            break

        try:
            tensor_input = encode_sequence(
                user_input).unsqueeze(0)
            if tensor_input.shape[1] == 0:
                print("Error: None of these words are in my vocabulary.\n")
                continue

            with torch.no_grad():
                prediction = model(tensor_input).item()

            if prediction > 0.5:
                print(
                    f" ALERT: SPAM DETECTED (Confidence: {prediction*100:.1f}%)\n")
            else:
                print(
                    f" SAFE: Normal Message (Confidence: {(1-prediction)*100:.1f}%)\n")

        except Exception as e:
            print(f"Error processing sequence: {e}\n")


test_rnn()
