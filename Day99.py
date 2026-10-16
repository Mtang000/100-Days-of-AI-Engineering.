import torch
import torch.nn.functional as F


class RingAttentionSimulator:
    def __init__(self, num_devices: int = 4):
        self.num_devices = num_devices

    def forward_pass(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        batch_size, total_seq_len, head_dim = Q.shape
        chunk_size = total_seq_len // self.num_devices
        device_outputs = []

        for rank in range(self.num_devices):
            q_local = Q[:, rank * chunk_size: (rank + 1) * chunk_size, :]

            acc_out = torch.zeros_like(q_local)
            running_max = torch.full(
                (batch_size, chunk_size, 1), -float('inf'), device=Q.device)
            running_sum = torch.zeros(
                (batch_size, chunk_size, 1), device=Q.device)

            for step in range(self.num_devices):
                kv_source_idx = (rank - step) % self.num_devices
                k_chunk = K[:, kv_source_idx *
                            chunk_size: (kv_source_idx + 1) * chunk_size, :]
                v_chunk = V[:, kv_source_idx *
                            chunk_size: (kv_source_idx + 1) * chunk_size, :]

                scores = torch.matmul(
                    q_local, k_chunk.transpose(-2, -1)) / (head_dim ** 0.5)

                block_max = scores.max(dim=-1, keepdim=True)[0]
                new_max = torch.maximum(running_max, block_max)

                exp_scores = torch.exp(scores - new_max)
                correction = torch.exp(running_max - new_max)

                running_sum = running_sum * correction + \
                    exp_scores.sum(dim=-1, keepdim=True)

                acc_out = acc_out * correction + \
                    torch.matmul(exp_scores, v_chunk)
                running_max = new_max

            device_outputs.append(acc_out / running_sum)

        return torch.cat(device_outputs, dim=1)


if __name__ == "__main__":
    torch.manual_seed(42)

    num_devices = 4
    batch_size = 1
    total_seq_len = 64
    head_dim = 64

    Q = torch.randn(batch_size, total_seq_len, head_dim)
    K = torch.randn(batch_size, total_seq_len, head_dim)
    V = torch.randn(batch_size, total_seq_len, head_dim)

    ring_engine = RingAttentionSimulator(num_devices=num_devices)
    output = ring_engine.forward_pass(Q, K, V)

    print(f"Total Sequence Length:     {total_seq_len}")
    print(f"Virtual Devices (Shards):  {num_devices}")
    print(
        f"Shard Length per Device:   {total_seq_len // num_devices}")
    print(f"Output Tensor Shape:       {output.shape}")
