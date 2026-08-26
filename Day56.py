import torch

vocab = {0: "The", 1: "dog", 2: "barked", 3: "meowed", 4: "loudly"}

current_sentence = [0]


def fast_draft_model(sentence):
    return [1, 3, 4]


def slow_target_model(sentence_plus_draft):
    true_predictions = [1, 2, 4]

    return true_predictions


print(f"Current Sentence: '{vocab[current_sentence[0]]}'\n")
print("--- Cycle 1 ---")

drafted_tokens = fast_draft_model(current_sentence)
draft_text = " ".join([vocab[t] for t in drafted_tokens])
print(f"1. Mini AI drafts 3 words: '{draft_text}'")

combined_sequence = current_sentence + drafted_tokens
true_tokens = slow_target_model(combined_sequence)

accepted_tokens = []
for i in range(len(drafted_tokens)):
    draft_word = drafted_tokens[i]
    smart_word = true_tokens[i]

    if draft_word == smart_word:
        print(
            f"2. Massive AI confirms: '{vocab[draft_word]}' (Accepted)")
        accepted_tokens.append(draft_word)
    else:
        print(
            f"2. Massive AI rejects: '{vocab[draft_word]}' is wrong Correcting to '{vocab[smart_word]}'.")
        accepted_tokens.append(smart_word)
        break
current_sentence.extend(accepted_tokens)

final_text = " ".join([vocab[t] for t in current_sentence])
print(f"\n--- Final Output ---")
print(f"Result: '{final_text}'")
