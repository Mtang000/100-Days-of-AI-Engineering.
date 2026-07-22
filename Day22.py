import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_length=100):
        super().__init__()
        pe = torch.zeros(max_length, embed_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_length = x.size(1)
        return x + self.pe[:, :seq_length, :]


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.pos_encoder = PositionalEncoding(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)

        self.output_layer = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):

        x = self.embedding(x)
        x = self.pos_encoder(x)

        x = self.transformer(x)

        return self.output_layer(x)


VOCAB_SIZE = 5000
EMBED_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 3

model = MiniGPT(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS
)

dummy_input = torch.randint(0, VOCAB_SIZE, (1, 10))

print(f"User Input Shape: {dummy_input.shape} (1 Batch, 10 Words)")


with torch.no_grad():
    predictions = model(dummy_input)

print(
    f"Output Shape: {predictions.shape} (1 Batch, 10 Words, 5000 Vocab Probabilities)\n")
print("Engine Online. The architecture successfully processed the sequence and generated next-word predictions!")
