import torch
import torch.nn as nn
import torch.optim as optim


model = nn.Linear(100, 10)
optimizer = optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

inputs = torch.randn(32, 100)
targets = torch.randn(32, 10)

scaler = torch.amp.GradScaler('cuda')

print("--- Starting Training Loop ---\n")

optimizer.zero_grad()

with torch.amp.autocast('cuda'):
    print("[1] Forward Pass running in fast FP16 mode...")
    predictions = model(inputs)
    loss = loss_fn(predictions, targets)

print("[2] Backward Pass running in fast FP16 mode...")
scaler.scale(loss).backward()

print("[3] Updating weights safely in highly precise FP32 mode...")
scaler.step(optimizer)
scaler.update()
