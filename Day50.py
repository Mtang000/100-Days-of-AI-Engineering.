import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.position = nn.Parameter(torch.randn(1, 100, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.transformer_blocks = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)

        self.to_vocab = nn.Linear(embed_dim, vocab_size)

    def forward(self, token_ids):
        seq_length = token_ids.shape[1]

        x = self.embedding(token_ids) + self.position[:, :seq_length, :]

        mask = nn.Transformer.generate_square_subsequent_mask(seq_length)

        x = self.transformer_blocks(x, is_causal=True, mask=mask)

        logits = self.to_vocab(x)
        return logits


vocab = {0: "I", 1: "went", 2: "to", 3: "the", 4: "store",
         5: "park", 6: "and", 7: "bought", 8: "apples", 9: "."}
reverse_vocab = {v: k for k, v in vocab.items()}

ai_model = MiniTransformer(
    vocab_size=10, embed_dim=16, num_heads=2, num_layers=2)
ai_model.eval()

current_sentence = torch.tensor([[0, 1, 2, 3]])
print("Input: I went to the")
print("\n--- Generating Text ---")

for step in range(4):
    with torch.no_grad():

        predictions = ai_model(current_sentence)

        next_word_logits = predictions[0, -1, :]

        next_word_id = torch.argmax(next_word_logits).item()

        word_text = vocab[next_word_id]
        print(f"Generated Word {step+1}: {word_text}")

        new_token = torch.tensor([[next_word_id]])
        current_sentence = torch.cat([current_sentence, new_token], dim=1)

print("\nFinal Output Sequence IDs:")
print(current_sentence.tolist())
