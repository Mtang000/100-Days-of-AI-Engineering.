import torch
import torch.nn.functional as F

print("--- Something for the title ---\n")

vocabulary = ["survive", "glitch", "tape",
              "run", "win", "sleep", "open", "play"]


raw_scores = torch.tensor([8.5, 7.0, 3.0, 2.5, 1.0, 4.0, 3.5, 2.0])


def generate_next_word(temperature=1.0):

    adjusted_scores = raw_scores / temperature

    probabilities = F.softmax(adjusted_scores, dim=0)

    winning_index = torch.multinomial(probabilities, num_samples=1).item()

    return vocabulary[winning_index], probabilities


print("TEST 1: Very Low Temperature (T = 0.1) -> The Robot")
print("The AI becomes incredibly strict and almost always picks the #1 safest word.\n")
for i in range(3):
    word, probs = generate_next_word(temperature=0.1)

    print(f"Attempt {i+1}: 'How to {word}'")

print("-" * 50)

print("TEST 2: Normal Temperature (T = 1.0) -> The Standard AI")
print("The AI uses its normal confidence. It usually picks top words, but occasionally takes a small risk.\n")
for i in range(3):
    word, probs = generate_next_word(temperature=1.0)
    print(f"Attempt {i+1}: 'How to {word}'")

print("-" * 50)

print("TEST 3: Very High Temperature (T = 3.0) -> Maximum Chaos")
print("The percentages flatten out. Every word has an almost equal chance of being picked.\n")
for i in range(3):
    word, probs = generate_next_word(temperature=3.0)
    print(f"Attempt {i+1}: 'How to {word}'")
