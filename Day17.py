import torch
import torch.nn as nn


embedding_dict = {
    "screen":    torch.tensor([0.90, -0.45,  0.10]),
    "display":   torch.tensor([0.88, -0.40,  0.15]),
    "shattered": torch.tensor([-0.50,  0.80, -0.20]),
    "broken":    torch.tensor([-0.45,  0.85, -0.10]),
    "apple":     torch.tensor([0.10,  0.10,  0.90]),
    "banana":    torch.tensor([0.15,  0.05,  0.85])
}


def get_vector(word):
    """Fetches the vector for a word, or returns zeroes if unknown."""
    word = word.lower()
    if word in embedding_dict:
        return embedding_dict[word]
    else:

        return torch.tensor([0.0, 0.0, 0.0])


cosine_sim = nn.CosineSimilarity(dim=0)


def compare_words(word1, word2):
    vec1 = get_vector(word1)
    vec2 = get_vector(word2)

    score = cosine_sim(vec1, vec2).item()

    match_pct = (score + 1) / 2 * 100

    print(f"Comparing '{word1}' and '{word2}':")
    print(f"Similarity Score: {match_pct:.1f}%")

    if match_pct > 90:
        print("Verdict: Basically the same word (Synonyms)\n")
    elif match_pct > 60:
        print("Verdict: Somewhat related\n")
    else:
        print("Verdict: Completely unrelated\n")


compare_words("screen", "display")

compare_words("broken", "shattered")

compare_words("screen", "banana")

compare_words("apple", "banana")
