import torch
import torch.nn as nn


class WorkerAgent(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, task_context, feedback_context=None):
        if feedback_context is not None:
            combined_input = task_context + feedback_context
        else:
            combined_input = task_context

        return self.generator(combined_input)


class CriticAgent(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.evaluator = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid())
        self.feedback_generator = nn.Linear(embed_dim, embed_dim)

    def forward(self, task_context, worker_output):
        combined_context = torch.cat([task_context, worker_output], dim=1)

        score = self.evaluator(combined_context)

        feedback = self.feedback_generator(worker_output)

        return score, feedback


embed_dim = 64
worker = WorkerAgent(embed_dim)
critic = CriticAgent(embed_dim)

print("Task: 'Write a secure login function in Python.'\n")
user_task = torch.randn(1, embed_dim)

passing_threshold = 0.85
max_attempts = 5

current_feedback = None
success = False

for attempt in range(1, max_attempts + 1):
    print(f"--- ATTEMPT {attempt} ---")

    generated_solution = worker(user_task, current_feedback)

    score, current_feedback = critic(user_task, generated_solution)
    score_value = score.item()

    print(f"Critic Score: {score_value * 100:.1f}%")

    if score_value >= passing_threshold:
        print("✅ Critic Approved. Outputting final result to user.\n")
        success = True
        break
    else:
        print("❌ Critic Rejected. Routing feedback tensor back to Worker.\n")

if not success:
    print(
        f"System halted. Worker failed to meet the {passing_threshold * 100}% threshold after {max_attempts} attempts.")
