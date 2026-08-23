import torch
import torch.nn as nn
import torch.optim as optim
model = nn.Linear(10, 2)
optimizer = optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

effective_batch_size = 128
micro_batch_size = 16

accumulation_steps = effective_batch_size // micro_batch_size

print(f"Target Batch Size: {effective_batch_size}")
print(f"Hardware Limit (Micro-Batch): {micro_batch_size}")
print(
    f"We will accumulate gradients for {accumulation_steps} steps before updating.\n")

optimizer.zero_grad()

for step in range(1, accumulation_steps + 1):
    dummy_inputs = torch.randn(micro_batch_size, 10)
    dummy_targets = torch.randn(micro_batch_size, 2)

    predictions = model(dummy_inputs)

    loss = loss_fn(predictions, dummy_targets) / accumulation_steps

    loss.backward()

    print(
        f"Step {step}: Processed {micro_batch_size} items. . GPU memory cleared.")

print("\n Reached 128 items")

optimizer.step()
optimizer.zero_grad()
print("Brain updated successfully")
