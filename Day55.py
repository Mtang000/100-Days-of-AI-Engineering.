import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os


def setup_process(rank, world_size):
    """Initializes the communication network between GPUs."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


rank = 0
world_size = 2
print(f"[GPU {rank}] Waking up and connecting to the network...")
setup_process(rank, world_size)

model = nn.Linear(10, 2)

ddp_model = DDP(model)

if rank == 0:
    data = torch.randn(16, 10)
else:
    data = torch.randn(16, 10)

target = torch.randn(16, 2)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)

output = ddp_model(data)
loss = loss_fn(output, target)


loss.backward()


print(f"[GPU {rank}] Updating weights...")
optimizer.step()

print(f"\n[GPU {rank}] Success.")
cleanup()
