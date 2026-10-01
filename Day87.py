import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedCrossAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.gate_alpha = nn.Parameter(torch.tensor([0.0]))

    def forward(self, text_x: torch.Tensor, image_x: torch.Tensor) -> torch.Tensor:
        batch_size, text_seq_len, d_model = text_x.shape
        _, image_seq_len, _ = image_x.shape

        q = self.q_proj(text_x).view(batch_size, text_seq_len,
                                     self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(image_x).view(batch_size, image_seq_len,
                                      self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(image_x).view(batch_size, image_seq_len,
                                      self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)

        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, text_seq_len, d_model)
        attn_output = self.out_proj(attn_output)

        gate_value = torch.tanh(self.gate_alpha)

        output = text_x + (gate_value * attn_output)

        return output, gate_value


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 2
    d_model = 512

    text_embeddings = torch.randn(batch_size, 10, d_model)
    image_embeddings = torch.randn(batch_size, 256, d_model)

    gated_cross_attn = GatedCrossAttention(d_model=d_model, num_heads=8)

    output_untrained, current_gate = gated_cross_attn(
        text_embeddings, image_embeddings)

    print(f"Gate Multiplier at step 0: {current_gate.item():.4f}")

    difference = torch.sum(
        torch.abs(output_untrained - text_embeddings)).item()
    print(f"Difference between output and raw text: {difference}")

    gated_cross_attn.gate_alpha.data = torch.tensor([0.5])

    output_trained, trained_gate = gated_cross_attn(
        text_embeddings, image_embeddings)
    print(f"Gate Multiplier after training: {trained_gate.item():.4f}")

    difference_trained = torch.sum(
        torch.abs(output_trained - text_embeddings)).item()
    print(
        f"Difference after training: {difference_trained:.2f}")
