import torch
import torch.nn.functional as F

vocab = {
    0: "{\"",       # JSON start and key quote
    1: "name",      # Key
    2: "\": \"",    # Key-value separator
    3: "Alice",     # String value
    4: "\", \"",    # Next item separator
    5: "age",       # Key
    6: "\": ",      # Key-value separator for numbers
    7: "25",        # Number value
    8: "}",         # JSON end
    9: "Hello",     # Standard text
    10: "world"     # Standard text
}
vocab_size = len(vocab)


def apply_syntax_mask(raw_logits, current_state):
    allowed_tokens_mask = torch.zeros_like(raw_logits, dtype=torch.bool)

    if current_state == "START":
        allowed_tokens_mask[0, 0] = True

    elif current_state == "NEEDS_KEY":
        allowed_tokens_mask[0, 1] = True
        allowed_tokens_mask[0, 5] = True

    elif current_state == "NEEDS_STRING_VALUE":
        allowed_tokens_mask[0, 3] = True

    constrained_logits = raw_logits.masked_fill(
        ~allowed_tokens_mask, float('-inf'))

    return constrained_logits


raw_model_logits = torch.tensor(
    [[1.2, 5.4, -0.2, 3.1, 0.4, 6.2, 1.1, 2.0, 0.1, 8.5, 4.3]])

print(
    f"\nModel's Top Choice Without Mask: '{vocab[torch.argmax(raw_model_logits).item()]}'")

constrained_logits = apply_syntax_mask(
    raw_model_logits, current_state="NEEDS_KEY")

probabilities = F.softmax(constrained_logits, dim=-1)

selected_token_id = torch.argmax(probabilities).item()
selected_token_text = vocab[selected_token_id]

print(f"Model's Top Choice WITH Mask: '{selected_token_text}'")

print("\nFinal Probability Distribution")
for i in range(vocab_size):
    prob = probabilities[0, i].item() * 100
    if prob > 0:
        print(f"Token [{vocab[i]:<6}] : {prob:>6.2f}%")
