import torch
import torch.optim as optim
from train_nsa_model import NSALanguageModel, NSAModelConfig
import time

print("Starting full training loop test...")
config = NSAModelConfig(
    vocab_size=128,
    hidden_dim=32,
    head_dim=16,
    n_heads=2,
    n_kv_heads=2,
    n_layers=1,
    seq_len=64,
    block_size=8,
    selected_blocks=2
)
model = NSALanguageModel(config)
model.train()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

input_ids = torch.randint(0, config.vocab_size, (1, config.seq_len))
targets = torch.randint(0, config.vocab_size, (1, config.seq_len))

t0 = time.time()
optimizer.zero_grad()
logits, loss = model(input_ids, targets=targets, kernel_fn=None)
loss.backward()
optimizer.step()
t1 = time.time()

print(f"Loss: {loss.item()}")
print(f"Training step time: {t1 - t0:.2f}s")
