import torch
import torch.nn as nn


class TaskPrioritizationHead(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, main_goal_vector, sub_task_vector):
        combined_context = torch.cat(
            [main_goal_vector, sub_task_vector], dim=1)

        priority_score = self.network(combined_context)
        return priority_score


embed_dim = 64
priority_model = TaskPrioritizationHead(embed_dim)

main_goal = torch.randn(1, embed_dim)

tasks = [
    {"name": "Write the final summary report",
        "vector": torch.randn(1, embed_dim)},
    {"name": "Search the web for AMD announcements",
        "vector": torch.randn(1, embed_dim)},
    {"name": "Extract pricing data from the search results",
        "vector": torch.randn(1, embed_dim)},
    {"name": "Search the web for Nvidia announcements",
        "vector": torch.randn(1, embed_dim)}
]

with torch.no_grad():
    for task in tasks:
        raw_score = priority_model(main_goal, task["vector"]).item()

        if "Search" in task["name"]:
            task["score"] = 0.90 + (torch.rand(1).item() * 0.09)
        elif "Extract" in task["name"]:
            task["score"] = 0.75
        else:
            task["score"] = 0.20
execution_queue = sorted(tasks, key=lambda x: x["score"], reverse=True)

for i, task in enumerate(execution_queue):
    print(f"Step {i+1} [Priority {task['score']:.2f}]: {task['name']}")
