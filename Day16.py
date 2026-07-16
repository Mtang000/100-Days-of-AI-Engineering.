import torch
import torch.nn as nn
import torch.optim as optim

training_data = [
    ("my screen is broken", 0),
    ("how do i reset my password", 0),
    ("the app keeps crashing", 0),

    ("how much does the enterprise plan cost", 1),
    ("do you offer student discounts", 1),
    ("i want to buy a subscription", 1),

    ("i need a refund immediately", 2),
    ("you overcharged my credit card", 2),
    ("where is my invoice", 2)
]


vocabulary = set()
for sentence, _ in training_data:
    for word in sentence.split():
        vocabulary.add(word)

vocab_list = list(vocabulary)
vocab_size = len(vocab_list)


def encode_sentence(sentence):
    word_counts = [0] * vocab_size
    words = sentence.lower().split()
    for word in words:
        if word in vocab_list:
            index = vocab_list.index(word)
            word_counts[index] += 1
    return torch.tensor(word_counts, dtype=torch.float32)


X_train = torch.stack([encode_sentence(text) for text, _ in training_data])
y_train = torch.tensor([label for _, label in training_data], dtype=torch.long)


class IntentRouter(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, 16)

        self.output_layer = nn.Linear(16, 3)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        raw_outputs = self.output_layer(x)
        return self.softmax(raw_outputs)


model = IntentRouter(vocab_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)

epochs = 200
for epoch in range(epochs):
    predictions = model(X_train)
    loss = criterion(predictions, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

department_names = {0: "🛠️ TECH SUPPORT", 1: "💰 SALES", 2: "🧾 BILLING"}


def run_chatbot():
    print("--- Customer Support Portal ---")
    print("Type a message (e.g. 'I need to buy a plan' or 'Refund me'). Type 'quit' to exit.\n")

    while True:
        user_input = input("Customer: ")
        if user_input.lower() == 'quit':
            break

        tensor_input = encode_sentence(user_input).unsqueeze(0)

        with torch.no_grad():
            probabilities = model(tensor_input)[0]

        confidence, predicted_class = torch.max(probabilities, dim=0)

        dept = department_names[predicted_class.item()]
        conf_pct = confidence.item() * 100

        print(
            f" AI Router: Transferring to {dept} (Confidence: {conf_pct:.1f}%)\n")


run_chatbot()
