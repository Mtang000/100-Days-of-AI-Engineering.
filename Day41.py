import torch
import torch.nn as nn
import torch.optim as optim

print("--- Before ---\n")


class MockLLM(nn.Module):
    def __init__(self):
        super().__init__()

        self.brain = nn.Linear(10, 3)

    def forward(self, x):
        return self.brain(x)


class RewardModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.judge = nn.Linear(10, 1)

    def forward(self, x):
        return self.judge(x)


main_ai = MockLLM()
reward_model = RewardModel()


optimizer = optim.Adam(main_ai.parameters(), lr=0.1)


context = torch.randn(1, 10)

print("User: 'How do I steal a car?'\n")


action_logits = main_ai(context)
action_probs = torch.softmax(action_logits, dim=-1)

print("Main AI's initial thoughts (Word Probabilities):")
print(f"Word A (Helpful): {action_probs[0][0].item()*100:.1f}%")
print(f"Word B (Toxic):   {action_probs[0][1].item()*100:.1f}%")
print(f"Word C (Neutral): {action_probs[0][2].item()*100:.1f}%\n")


mock_reward = torch.tensor([[-5.0]])

print(f"Reward Model Score: {mock_reward.item():.1f} (Very Bad)\n")


log_probs = torch.log_softmax(action_logits, dim=-1)

chosen_action_log_prob = log_probs[0][1]


rl_loss = -(chosen_action_log_prob * mock_reward.item())

optimizer.zero_grad()
rl_loss.backward()
optimizer.step()

print("--- After ---\n")


new_action_logits = main_ai(context)
new_action_probs = torch.softmax(new_action_logits, dim=-1)

print("Main AI's new thoughts after being punished:")
print(f"Word A (Helpful): {new_action_probs[0][0].item()*100:.1f}%")
print(
    f"Word B (Toxic):   {new_action_probs[0][1].item()*100:.1f}% (Drastically dropped!)")
print(f"Word C (Neutral): {new_action_probs[0][2].item()*100:.1f}%")
