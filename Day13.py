import torch
import torch.nn as nn
import torch.optim as optim

reviews = [
    "I love this movie it is amazing",  # Positive (1)
    "This movie is terrible and broken",  # Negative (0)
    "Great movie I love it",            # Positive (1)
    "I hate this terrible movie",       # Negative (0)
]
labels = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
vocabulary = set()
for review in reviews:
    for word in review.lower().split():
        vocabulary.add(word)

vocab_list = list(vocabulary)
vocab_size = len(vocab_list)

print(f"Vocabulary Size: {vocab_size} unique words.\n")


def encode_sentence(sentence):
    """
    Converts a sentence into a mathematical tensor.
    If a word from the vocab is in the sentence, it gets a 1. If not, a 0.
    """
    word_counts = [0] * vocab_size
    words = sentence.lower().split()

    for word in words:
        if word in vocab_list:
            index = vocab_list.index(word)
            word_counts[index] += 1

    return torch.tensor(word_counts, dtype=torch.float32)


X_data = torch.stack([encode_sentence(r) for r in reviews])


class SentimentBrain(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layer1 = nn.Linear(in_features=input_size, out_features=8)
        self.output_layer = nn.Linear(in_features=8, out_features=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.sigmoid(self.output_layer(x))
        return x


model = SentimentBrain(input_size=vocab_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)


epochs = 100
for epoch in range(epochs):
    predictions = model(X_data)
    loss = criterion(predictions, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def test_ai():
    print("Type a short movie review using words like: love, hate, amazing, terrible, great, broken.")
    print("Type 'quit' to exit.\n")

    while True:
        user_input = input("Enter your review: ")
        if user_input.lower() == 'quit':
            break

        tensor_input = encode_sentence(user_input)

        with torch.no_grad():
            prediction = model(tensor_input).item()

        if prediction > 0.5:
            print(
                f"AI Verdict: POSITIVE (Confidence: {prediction*100:.1f}%)\n")
        else:
            print(
                f"AI Verdict: NEGATIVE (Confidence: {(1-prediction)*100:.1f}%)\n")


test_ai()
