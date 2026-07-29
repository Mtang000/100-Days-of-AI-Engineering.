import torch
import torch.nn as nn
import torch.optim as optim

print("--- Retrieval-Augmented Generation ---\n")

database = {
    "Selfie tips": "To improve your selfies, try different angles and lighting.",
    "Improve handwriting": "Practice regularly and focus on letter formation.",
    "Arm pain": "Use ice packs and avoid repetitive motions."
}


def search_database(user_query):
    print(f"[System] Searching database for keywords in: '{user_query}'...")
    for keyword, fact in database.items():
        if keyword.lower() in user_query.lower():
            print(f"[System] Match found! Retrieving fact: '{fact}'")
            return fact
    print("[System] No matches found.")
    return "I don't have any information on that topic."


chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,:;!?<>[]-'\n"
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}


class MockAssistant(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 32)
        self.brain = nn.Linear(32, vocab_size)

    def forward(self, x):
        return self.brain(self.embedding(x))


assistant_model = MockAssistant(vocab_size)


def chat_with_rag(user_prompt):
    print("\n--- AI Assistant ---")
    print(f"User: {user_prompt}")

    retrieved_fact = search_database(user_prompt)

    augmented_prompt = f"<SYSTEM> Use this fact to answer: {retrieved_fact} </SYSTEM> <USER> {user_prompt} <ASSISTANT>"

    print(f"\n[Behind the Scenes] The augmented prompt sent to the AI is:")
    print(augmented_prompt)

    print("\nAI Response:")
    generated_text = f"Based on my knowledge base, {retrieved_fact.lower()}"
    print(generated_text)
    print("-" * 50)


print("Type 'quit' to exit the chat.\n")

while True:

    user_input = input("Ask a question: ")

    if user_input.lower() == 'quit':
        print("Shutting down assistant...")
        break

    chat_with_rag(user_input)
