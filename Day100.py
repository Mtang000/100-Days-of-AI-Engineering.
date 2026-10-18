import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralAgentWorkspace(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

        self.memory_cell = nn.GRUCell(d_model, d_model)

        self.termination_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

        self.action_head = nn.Linear(d_model, d_model)

    def forward(
        self,
        current_input: torch.Tensor,
        scratchpad_state: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = current_input.shape[0]

        if scratchpad_state is None:
            scratchpad_state = torch.zeros(
                batch_size, self.d_model, device=current_input.device)

        next_state = self.memory_cell(current_input, scratchpad_state)

        stop_probability = self.termination_gate(next_state)

        action_vector = self.action_head(next_state)

        return action_vector, next_state, stop_probability


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 2
    d_model = 512

    agent_workspace = NeuralAgentWorkspace(d_model=d_model)

    query_embed = torch.randn(batch_size, d_model)

    action_1, state_1, stop_p_1 = agent_workspace(
        query_embed, scratchpad_state=None)

    print(
        f"Step 1 - Stop Probability: {stop_p_1.squeeze().tolist()} ")

    tool_observation = torch.randn(batch_size, d_model)

    action_2, state_2, stop_p_2 = agent_workspace(
        tool_observation, scratchpad_state=state_1)

    print(
        f"Step 2 - Stop Probability: {stop_p_2.squeeze().tolist()} ")
    print(f"Final Action Output Shape: {action_2.shape}")
