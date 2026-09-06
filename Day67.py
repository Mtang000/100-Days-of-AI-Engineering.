import torch
import torch.nn as nn


class CrossEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.score_classifier = nn.Linear(embed_dim, 1)

    def forward(self, query_tokens, document_tokens):
        batch_size = query_tokens.shape[0]

        sep_token = torch.zeros(batch_size, 1, dtype=torch.long)
        combined_sequence = torch.cat(
            [query_tokens, sep_token, document_tokens], dim=1)

        embeddings = self.embedding(combined_sequence)

        transformer_output = self.transformer(embeddings)

        sequence_representation = transformer_output[:, 0, :]

        match_score = self.score_classifier(sequence_representation)

        return match_score


vocab_size = 5000
embed_dim = 128
model = CrossEncoder(vocab_size, embed_dim)

query = torch.randint(1, vocab_size, (1, 10))

doc1 = torch.randint(1, vocab_size, (1, 50))
doc2 = torch.randint(1, vocab_size, (1, 50))
doc3 = torch.randint(1, vocab_size, (1, 50))


score1 = model(query, doc1).item()
score2 = model(query, doc2).item()
score3 = model(query, doc3).item()

print(f"Document 1 Re-Rank Score: {score1:.4f}")
print(f"Document 2 Re-Rank Score: {score2:.4f}")
print(f"Document 3 Re-Rank Score: {score3:.4f}")

scores = [score1, score2, score3]
best_doc = scores.index(max(scores)) + 1

print(
    f"\nCross-Encoder identified Document {best_doc} as the most accurate match.")
