import torch
import torch.nn as nn
import torch.optim as optim


entities = {"User": 0, "Buster": 1, "Chicken": 2}
relations = {"owns": 0, "hates": 1}

subject = torch.tensor([entities["Buster"]])
relation = torch.tensor([relations["hates"]])
target_object = torch.tensor([entities["Chicken"]])


class KnowledgeGraphMemory(nn.Module):
    def __init__(self, num_entities, num_relations, embed_dim=16):
        super().__init__()

        self.entity_embeddings = nn.Embedding(num_entities, embed_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embed_dim)

    def forward(self, sub, rel):

        sub_vec = self.entity_embeddings(sub)
        rel_vec = self.relation_embeddings(rel)

        predicted_obj_vec = sub_vec + rel_vec
        return predicted_obj_vec


ai_memory = KnowledgeGraphMemory(len(entities), len(relations))
optimizer = optim.Adam(ai_memory.parameters(), lr=0.1)

print("Fact: [Buster] -> [hates] -> [Chicken]")


for epoch in range(50):
    optimizer.zero_grad()

    predicted_vec = ai_memory(subject, relation)

    actual_obj_vec = ai_memory.entity_embeddings(target_object)

    loss = torch.norm(predicted_vec - actual_obj_vec, p=2)

    loss.backward()
    optimizer.step()


print("\n--- Memory Test ---")
ai_memory.eval()
with torch.no_grad():

    query_vec = ai_memory(subject, relation)

    print("Checking distances to all known entities:")
    for name, idx in entities.items():
        entity_vec = ai_memory.entity_embeddings(torch.tensor([idx]))
        distance = torch.norm(query_vec - entity_vec, p=2).item()

        if distance < 0.5:
            print(f"- {name}: Distance {distance:.4f}")
        else:
            print(f"- {name}: Distance {distance:.4f}")
