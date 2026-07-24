import torch
import torch.nn as nn
import torch.optim as optim
import math


corpus = "the dog barks at night and the cat sleeps on the mat"

chars = sorted(list(set(corpus)))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}


data = torch.tensor([char_to_idx[c] for c in corpus], dtype=torch.long)

print(f"Corpus Length: {len(corpus)} characters")
print(f"Vocabulary Size: {vocab_size} unique characters\n")


X = data[:-1].unsqueeze(0)

Y = data[1:].unsqueeze(0)


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, num_heads=2, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        pe = torch.zeros(100, embed_dim)
        position = torch.arange(0, 100, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        seq_len = x.size(1)

        x = self.embedding(x) + self.pe[:, :seq_len, :]

        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len).to(x.device)

        x = self.transformer(x, mask=causal_mask, is_causal=True)

        return self.output_layer(x)


model = MiniGPT(vocab_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("Training on the corpus...\n")
epochs = 300

for epoch in range(epochs):
    optimizer.zero_grad()

    logits = model(X)

    loss = criterion(logits.view(-1, vocab_size), Y.view(-1))

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")


def generate_text(start_str, max_new_chars=20):
    model.eval()
    context = [char_to_idx[c] for c in start_str]

    with torch.no_grad():
        for _ in range(max_new_chars):
            x = torch.tensor([context], dtype=torch.long)
            logits = model(x)

            next_char_logits = logits[0, -1, :]
            next_char_idx = torch.argmax(next_char_logits).item()

            context.append(next_char_idx)

    return "".join([idx_to_char[i] for i in context])


print("---  AI Text Generation Test ---")
generated_result = generate_text("the dog", max_new_chars=15)
print(f"Prompt: 'the dog'")
print(f"Generated: '{generated_result}'")
