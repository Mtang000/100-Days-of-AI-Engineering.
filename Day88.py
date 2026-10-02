import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.codebook = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.codebook.weight.data.uniform_(-1 /
                                           self.num_embeddings, 1 / self.num_embeddings)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat_inputs = inputs.reshape(-1, self.embedding_dim)

        distances = (
            torch.sum(flat_inputs ** 2, dim=1, keepdim=True)
            + torch.sum(self.codebook.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_inputs, self.codebook.weight.t())
        )

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        quantized = self.codebook(encoding_indices).view(inputs.shape)

        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())

        vq_loss = e_latent_loss + self.commitment_cost * q_latent_loss

        quantized = inputs + (quantized - inputs).detach()

        return quantized, vq_loss, encoding_indices.view(inputs.shape[0], inputs.shape[1])


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 2
    sequence_length = 16
    embedding_dim = 64
    codebook_size = 512   #
    continuous_features = torch.randn(
        batch_size, sequence_length, embedding_dim)
    continuous_features.requires_grad_(True)

    vq_layer = VectorQuantizer(
        num_embeddings=codebook_size, embedding_dim=embedding_dim)

    # Forward pass
    quantized_features, loss, token_ids = vq_layer(continuous_features)

    print(
        f"Input Shape:      {continuous_features.shape} (Continuous float32)")
    print(f"Token IDs Shape:  {token_ids.shape} (Discrete integers)")
    print(
        f"Quantized Shape:  {quantized_features.shape} (Snapped to codebook vectors)")
    print(f"\nVQ Loss: {loss.item():.4f}")
    loss.backward()
