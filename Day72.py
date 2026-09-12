import torch
import torch.nn as nn


class AgentMemoryManager(nn.Module):
    def __init__(self, embed_dim, context_limit):
        super().__init__()
        self.encoder = nn.Linear(embed_dim, embed_dim)
        self.context_limit = context_limit

        self.active_memory = []

        self.external_storage = []

    def process_and_add_message(self, message_text_tensor):
        if len(self.active_memory) >= self.context_limit:
            oldest_message_tensor = self.active_memory.pop(0)

            self.external_storage.append(oldest_message_tensor)
            print("-> Evicted oldest message to External Storage (Cold Storage).")
            print(
                f"-> Cold Storage now holds {len(self.external_storage)} tensors.")

        message_vector = self.encoder(message_text_tensor)
        self.active_memory.append(message_vector)
        print(
            f"-> Active memory now holds {len(self.active_memory)} tensors.\n")


embed_dim = 64
context_limit = 3

manager = AgentMemoryManager(embed_dim, context_limit)

msg1 = torch.randn(1, embed_dim)
manager.process_and_add_message(msg1)

msg2 = torch.randn(1, embed_dim)
manager.process_and_add_message(msg2)

msg3 = torch.randn(1, embed_dim)
manager.process_and_add_message(msg3)

msg4 = torch.randn(1, embed_dim)
manager.process_and_add_message(msg4)

msg5 = torch.randn(1, embed_dim)
manager.process_and_add_message(msg5)

stacked_context = torch.stack(manager.active_memory)
print(f"Final Input Tensor Shape (Active Memory): {stacked_context.shape}")
