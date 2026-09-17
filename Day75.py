import torch
import torch.nn as nn
import torch.nn.functional as F


class RouterAgent(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.classifier = nn.Linear(embed_dim, 4)

    def forward(self, current_context):
        logits = self.classifier(current_context)
        probabilities = F.softmax(logits, dim=-1)
        next_state = torch.argmax(probabilities, dim=-1).item()
        return next_state


embed_dim = 64
router = RouterAgent(embed_dim)
state_map = {
    0: "END",
    1: "RESEARCHER",
    2: "CODER",
    3: "REVIEWER"
}

state_vectors = [torch.randn(1, embed_dim) for _ in range(4)]
with torch.no_grad():
    # Force Researcher
    router.classifier.weight[1] = state_vectors[0].squeeze() * 10.0
    # Force Coder
    router.classifier.weight[2] = state_vectors[1].squeeze() * 10.0
    # Force Reviewer
    router.classifier.weight[3] = state_vectors[2].squeeze() * 10.0
    # Force End
    router.classifier.weight[0] = state_vectors[3].squeeze() * 10.0


step_count = 0
current_context = state_vectors[0]
active = True

while active:
    step_count += 1
    print(f"--- Graph Step {step_count} ---")

    next_node = router(current_context)
    node_name = state_map[next_node]
    print(f"Router Decision: Transitioning to [{node_name}]")

    if next_node == 0:
        print("Final Output sent to user.")
        active = False

    elif next_node == 1:
        print("[Researcher Agent executes]: Searching for website layout details...")
        current_context = state_vectors[1]

    elif next_node == 2:
        print("[Coder Agent executes]: Writing the Python scraping script...")
        current_context = state_vectors[2]

    elif next_node == 3:
        print("[Reviewer Agent executes]: Checking the code for errors. Code passes.")
        current_context = state_vectors[3]
