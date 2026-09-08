import torch
from collections import Counter


def get_ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def compute_bleu_precision(candidate_tokens, reference_tokens, n=1):
    cand_ngrams = get_ngrams(candidate_tokens, n)
    ref_ngrams = get_ngrams(reference_tokens, n)

    if not cand_ngrams:
        return 0.0

    cand_counts = Counter(cand_ngrams)
    ref_counts = Counter(ref_ngrams)

    clipped_matches = sum(
        min(count, ref_counts[ng]) for ng, count in cand_counts.items())
    precision = clipped_matches / len(cand_ngrams)
    return precision


def compute_rouge_l_recall(candidate_tokens, reference_tokens):
    m, n = len(candidate_tokens), len(reference_tokens)
    dp = torch.zeros((m + 1, n + 1), dtype=torch.int32)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if candidate_tokens[i - 1] == reference_tokens[j - 1]:
                dp[i, j] = dp[i - 1, j - 1] + 1
            else:
                dp[i, j] = torch.max(dp[i - 1, j], dp[i, j - 1])

    lcs_length = dp[m, n].item()
    recall = lcs_length / n if n > 0 else 0.0
    return recall


reference = "the quick brown fox jumps over the lazy dog"
candidate = "the fast brown fox leaped over the lazy dog"

ref_tokens = reference.split()
cand_tokens = candidate.split()

print(f"Reference Text: '{reference}'")
print(f"Candidate Text: '{candidate}'\n")

p1 = compute_bleu_precision(cand_tokens, ref_tokens, n=1)
p2 = compute_bleu_precision(cand_tokens, ref_tokens, n=2)

c_len = len(cand_tokens)
r_len = len(ref_tokens)
brevity_penalty = 1.0 if c_len > r_len else torch.exp(
    torch.tensor(1.0 - (r_len / c_len))).item()

bleu_score = brevity_penalty * (p1 * p2) ** 0.5
rouge_l_score = compute_rouge_l_recall(cand_tokens, ref_tokens)

print("--- Evaluation Results ---")
print(f"BLEU-1 Precision (Single words) : {p1 * 100:.1f}%")
print(f"BLEU-2 Precision (Word pairs)   : {p2 * 100:.1f}%")
print(f"Brevity Penalty                 : {brevity_penalty:.2f}")
print(f"Final Simplified BLEU Score     : {bleu_score:.4f}")
print(f"ROUGE-L Recall (LCS coverage)   : {rouge_l_score * 100:.1f}%")
