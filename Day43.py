import torch
import torch.nn as nn


sequence_length = 5
embed_dim = 16

state_dim = 32


x = torch.randn(sequence_length, embed_dim)


class SimpleSSM(nn.Module):
    def __init__(self):
        super().__init__()

        self.A = nn.Parameter(torch.randn(state_dim, state_dim))

        self.B = nn.Parameter(torch.randn(state_dim, embed_dim))

        self.C = nn.Parameter(torch.randn(embed_dim, state_dim))

    def forward(self, input_sequence):

        hidden_state = torch.zeros(state_dim)

        outputs = []

        for i in range(len(input_sequence)):
            current_word = input_sequence[i]

            hidden_state = torch.matmul(
                self.A, hidden_state) + torch.matmul(self.B, current_word)

            prediction = torch.matmul(self.C, hidden_state)
            outputs.append(prediction)

            print(f"Read word {i+1} -> Memory Box Size: {hidden_state.shape}")

        return torch.stack(outputs)


mamba_mock = SimpleSSM()


predictions = mamba_mock(x)

print(f"\nFinal Output Shape: {predictions.shape}")
print("The AI processed all 5 words without its memory box growing.")
