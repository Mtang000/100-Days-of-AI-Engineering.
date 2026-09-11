import torch
import torch.nn as nn
import torch.nn.functional as F


class ReActAgent(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.decision_head = nn.Linear(embed_dim, 3)

    def forward(self, context_state):
        logits = self.decision_head(context_state)
        probabilities = F.softmax(logits, dim=-1)
        decision_index = torch.argmax(probabilities, dim=-1).item()
        return decision_index, probabilities


embed_dim = 64
agent = ReActAgent(embed_dim)

print("User Prompt: 'Find the stock price of Apple, and multiply it by 10.'\n")

state_step1 = torch.randn(1, embed_dim)
with torch.no_grad():
    agent.decision_head.weight[1] = state_step1.squeeze() * 10.0

decision1, _ = agent(state_step1)
if decision1 == 1:
    print("AI Phase: THOUGHT")
    print("Output: 'I need to find the current Apple stock price first before I can multiply.'\n")

state_step2 = torch.randn(1, embed_dim)
with torch.no_grad():
    agent.decision_head.weight[2] = state_step2.squeeze() * 10.0

decision2, _ = agent(state_step2)
if decision2 == 2:
    print("AI Phase: ACTION")
    print("Output: <TOOL_CALL: get_stock_price(ticker='AAPL')>")
    print("[System]: Tool executed. Observation injected: 'Apple stock is 150 USD'.\n")

state_step3 = torch.randn(1, embed_dim)
with torch.no_grad():
    agent.decision_head.weight[0] = state_step3.squeeze() * 10.0

decision3, _ = agent(state_step3)
if decision3 == 0:
    print("AI Phase: FINAL ANSWER")
    print("Output: 'The current stock price of Apple is 150 USD, multiplying it by 10 gives 1500 USD.'\n")
