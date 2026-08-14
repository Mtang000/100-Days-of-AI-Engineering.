import torch
import torch.nn.functional as F


doc_1 = torch.tensor([0.8, 0.9, 0.1, 0.0])


doc_2 = torch.tensor([0.7, 0.8, 0.2, 0.1])

doc_3 = torch.tensor([0.1, 0.0, 0.9, 0.8])


vector_database = torch.stack([doc_1, doc_2, doc_3])

print("Database indexed: 3 documents stored as vectors.\n")


query = torch.tensor([0.8, 0.7, 0.1, 0.1])


def search_database(query_vector, database_matrix):

    similarity_scores = F.cosine_similarity(
        query_vector.unsqueeze(0), database_matrix)

    return similarity_scores


# Run the search
scores = search_database(query, vector_database)


print(f"\nDocument 1 (Adopt a cat):   Score: {scores[0].item():.4f}")
print(f"Document 2 (Feline food):   Score: {scores[1].item():.4f}")
print(f"Document 3 (Car tire):      Score: {scores[2].item():.4f}\n")

# Find the absolute best match
best_match_idx = torch.argmax(scores).item()
print(f"-> BEST MATCH: Document {best_match_idx + 1}")
