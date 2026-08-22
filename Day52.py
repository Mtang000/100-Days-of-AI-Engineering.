import torch
import torch.nn.functional as F


vocab = ["<start>", "I", "am", "happy", "coding", "a", "robot"]
vocab_size = len(vocab)
beam_width = 2


def mock_predict_next_word(current_word_idx):
    logits = torch.randn(vocab_size)
    return F.log_softmax(logits, dim=-1)


beams = [(0.0, [0])]

print("Starting Sentence: '<start>'\n")

for step in range(3):
    print(f"--- Step {step + 1} ---")
    all_candidates = []

    for score, sentence in beams:
        last_word = sentence[-1]

        log_probs = mock_predict_next_word(last_word)

        top_scores, top_indices = torch.topk(log_probs, beam_width)

        for i in range(beam_width):
            new_score = score + top_scores[i].item()
            new_sentence = sentence + [top_indices[i].item()]

            all_candidates.append((new_score, new_sentence))

            word_str = vocab[top_indices[i].item()]
            print(
                f"Candidate Branch: {vocab[last_word]} -> {word_str} (Total Score: {new_score:.2f})")

    all_candidates.sort(key=lambda x: x[0], reverse=True)

    beams = all_candidates[:beam_width]

    print("\n[PRUNING] Keeping only the Top 2 branches:")
    for b in beams:
        sentence_str = " ".join([vocab[idx] for idx in b[1]])
        print(f"- {sentence_str} (Score: {b[0]:.2f})")
    print("\n")

best_score, best_sentence = beams[0]
final_text = " ".join([vocab[idx] for idx in best_sentence])

print("--- 🏆 FINAL WINNER ---")
print(f"Highest probability path: '{final_text}'")
