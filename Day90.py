import torch
import torch.nn as nn
import torch.nn.functional as F

class SpeculativeEngine(nn.Module):
    def __init__(self, gamma: int = 4):
        super().__init__()
        self.gamma = gamma 
    def rejection_sampling(
        self, 
        draft_probs: torch.Tensor, 
        target_probs: torch.Tensor, 
        draft_tokens: torch.Tensor
    ) -> torch.Tensor:
        accepted_tokens = []
        
        for t in range(self.gamma):
            token_id = draft_tokens[t].item()
            q_val = draft_probs[t, token_id]
            p_val = target_probs[t, token_id]
            
            accept_prob = torch.minimum(
                torch.tensor(1.0, device=q_val.device), 
                p_val / (q_val + 1e-8)
            )
            u = torch.rand(1, device=q_val.device)

            if u.item() <= accept_prob.item():
                accepted_tokens.append(token_id)
            else:
                res_dist = F.relu(target_probs[t] - draft_probs[t])
                res_dist = res_dist / (res_dist.sum() + 1e-8)
                replacement_token = torch.multinomial(res_dist, num_samples=1).item()
                accepted_tokens.append(replacement_token)
                break

        if len(accepted_tokens) == self.gamma:
            extra_token = torch.multinomial(target_probs[self.gamma], num_samples=1).item()
            accepted_tokens.append(extra_token)

        return torch.tensor(accepted_tokens, dtype=torch.long)

if __name__ == "__main__":
    torch.manual_seed(42)

    gamma = 4
    vocab_size = 1000
    engine = SpeculativeEngine(gamma=gamma)

    draft_logits = torch.randn(gamma, vocab_size)
    draft_probs = F.softmax(draft_logits, dim=-1)
    draft_tokens = torch.argmax(draft_probs, dim=-1)  
    target_logits = torch.randn(gamma + 1, vocab_size)
    target_probs = F.softmax(target_logits, dim=-1)

    final_accepted_sequence = engine.rejection_sampling(
        draft_probs=draft_probs,
        target_probs=target_probs,
        draft_tokens=draft_tokens
    )
    print(f"Proposed Draft Tokens ({gamma}):      {draft_tokens.tolist()}")
    print(f"Accepted/Corrected Tokens ({len(final_accepted_sequence)}): {final_accepted_sequence.tolist()}")
    print(f"Accepted Yield Ratio:              {len(final_accepted_sequence)} tokens generated in 1 target pass.")
   
