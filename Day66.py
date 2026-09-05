import torch
import torch.nn as nn
import torch.nn.functional as F


class BiEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.document_encoder = nn.Embedding(vocab_size, embed_dim)
        self.query_encoder = nn.Embedding(vocab_size, embed_dim)

    def encode_document(self, doc_tokens):
        doc_vectors = self.document_encoder(doc_tokens)
        return torch.mean(doc_vectors, dim=1)

    def encode_query(self, query_tokens):
        query_vectors = self.query_encoder(query_tokens)
        return torch.mean(query_vectors, dim=1)


vocab_size = 1000
embed_dim = 128
model = BiEncoder(vocab_size, embed_dim)

doc1_tokens = torch.randint(0, vocab_size, (1, 50))
doc2_tokens = torch.randint(0, vocab_size, (1, 50))
doc3_tokens = torch.randint(0, vocab_size, (1, 50))

doc1_vec = model.encode_document(doc1_tokens)
doc2_vec = model.encode_document(doc2_tokens)
doc3_vec = model.encode_document(doc3_tokens)
database_matrix = torch.cat([doc1_vec, doc2_vec, doc3_vec], dim=0)

query_tokens = torch.randint(0, vocab_size, (1, 10))
query_vec = model.encode_query(query_tokens)


query_norm = F.normalize(query_vec, p=2, dim=1)
database_norm = F.normalize(database_matrix, p=2, dim=1)

similarity_scores = torch.matmul(query_norm, database_norm.transpose(0, 1))

scores = similarity_scores[0].tolist()
print(f"\nDocument 1 Match Score: {scores[0]:.4f}")
print(f"Document 2 Match Score: {scores[1]:.4f}")
print(f"Document 3 Match Score: {scores[2]:.4f}")

best_doc_index = torch.argmax(similarity_scores).item()
print(
    f"\nThe system retrieved Document {best_doc_index + 1} as the optimal match.")
