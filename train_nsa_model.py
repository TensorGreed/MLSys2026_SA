"""
train_nsa_model.py
==================
A fully functional mini-GPT language model that uses the DeepSeek
NativeSparseAttention from kernel.py as its core attention mechanism.

Usage:
    python train_nsa_model.py                       # Train on synthetic data (CPU/GPU)
    python train_nsa_model.py --data shakespeare    # Train on a text file

This file demonstrates:
    1. How to wrap NativeSparseAttention into a Transformer block
    2. How to build a full autoregressive language model
    3. How to compile the TileLang kernel (when CUDA is available)
    4. A complete training loop with loss logging
    5. Text generation at the end

Architecture:
    Token Embedding + Positional Embedding
    -> N x NSATransformerBlock(LayerNorm -> NSA -> Residual -> LayerNorm -> FFN -> Residual)
    -> LayerNorm -> Linear Head -> Logits
"""

import argparse
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Import our NSA implementation from kernel.py (same directory)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kernel import NativeSparseAttention

# Try to import the TileLang kernel compiler (only works on CUDA)
try:
    from kernel import native_sparse_attention as compile_nsa_kernel
    TILELANG_AVAILABLE = True
except Exception:
    TILELANG_AVAILABLE = False


# ===========================================================================
# Configuration
# ===========================================================================
class NSAModelConfig:
    """
    All hyperparameters in one place.

    Think of this as the "blueprint" for our model.
    Every number here controls a different aspect of the architecture.
    """
    # --- Model Architecture ---
    vocab_size: int = 256           # Character-level: 256 ASCII characters
    n_layers: int = 4               # Number of Transformer blocks stacked
    n_heads: int = 8                # Number of Query attention heads (HQ)
    n_kv_heads: int = 4             # Number of Key/Value heads (H) — GQA
    head_dim: int = 64              # Dimension per head (D)
    hidden_dim: int = 512           # = n_heads * head_dim (total model width)
    ffn_dim: int = 1024             # Feed-forward network intermediate size

    # --- NSA Specific ---
    block_size: int = 64            # Tokens per block for sparse attention
    selected_blocks: int = 16       # How many blocks the "Detective" reads
    compression_ratio: int = 4      # How much the "Skimmer" compresses (4x)
    window_size: int = 512          # How many recent tokens the "Secretary" sees

    # --- Training ---
    batch_size: int = 4
    seq_len: int = 1024             # Context window length
    learning_rate: float = 3e-4
    max_steps: int = 500
    eval_interval: int = 50         # Print loss every N steps
    weight_decay: float = 0.01
    dropout: float = 0.1
    grad_clip: float = 1.0

    # --- Generation ---
    gen_max_tokens: int = 200       # How many tokens to generate after training
    gen_temperature: float = 0.8

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise ValueError(f"Unknown config key: {k}")

        # Validate
        assert self.hidden_dim == self.n_heads * self.head_dim, \
            f"hidden_dim ({self.hidden_dim}) must equal n_heads * head_dim ({self.n_heads * self.head_dim})"
        assert self.n_heads % self.n_kv_heads == 0, \
            f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"


# ===========================================================================
# Feed-Forward Network (the "thinking" part of each Transformer block)
# ===========================================================================
class FeedForward(nn.Module):
    """
    A simple two-layer MLP with GELU activation.

    After the attention layer gathers relevant information,
    the FFN processes and transforms that information.

    Architecture: Linear(hidden -> ffn) -> GELU -> Dropout -> Linear(ffn -> hidden)
    """
    def __init__(self, config: NSAModelConfig):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_dim, config.ffn_dim, bias=False)
        self.w2 = nn.Linear(config.ffn_dim, config.hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, hidden_dim]
        return self.dropout(self.w2(F.gelu(self.w1(x))))


# ===========================================================================
# QKV Projection Layer
# ===========================================================================
class QKVProjection(nn.Module):
    """
    Projects the input hidden states into Q, K, V tensors.

    In GQA (Grouped Query Attention):
      - Q has more heads (n_heads) than K/V (n_kv_heads)
      - This saves memory on K/V while keeping Q expressive

    Input:  [B, T, hidden_dim]
    Output: Q [B, T, n_heads, head_dim], K [B, T, n_kv_heads, head_dim], V [same as K]
    """
    def __init__(self, config: NSAModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim

        # Separate projections for Q, K, V
        self.wq = nn.Linear(config.hidden_dim, config.n_heads * config.head_dim, bias=False)
        self.wk = nn.Linear(config.hidden_dim, config.n_kv_heads * config.head_dim, bias=False)
        self.wv = nn.Linear(config.hidden_dim, config.n_kv_heads * config.head_dim, bias=False)

        # Output projection: merges all heads back to hidden_dim
        self.wo = nn.Linear(config.n_heads * config.head_dim, config.hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor):
        B, T, _ = x.shape

        # Project and reshape into multi-head format
        Q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        K = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        V = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        return Q, K, V


# ===========================================================================
# Transformer Block using NSA
# ===========================================================================
class NSATransformerBlock(nn.Module):
    """
    A single Transformer block that uses NativeSparseAttention.

    Architecture (Pre-Norm style, like LLaMA/DeepSeek):
        x -> LayerNorm -> QKV Projection -> NSA -> Output Projection -> + x (residual)
          -> LayerNorm -> FFN -> + x (residual)

    Why Pre-Norm?
        Pre-Norm (LayerNorm before attention) trains more stably than
        Post-Norm (LayerNorm after attention), especially for deep models.
    """
    def __init__(self, config: NSAModelConfig, nsa_module: NativeSparseAttention):
        super().__init__()

        # Layer norms (Pre-Norm style)
        self.ln1 = nn.LayerNorm(config.hidden_dim)
        self.ln2 = nn.LayerNorm(config.hidden_dim)

        # Attention components
        self.qkv_proj = QKVProjection(config)
        self.nsa = nsa_module  # The NativeSparseAttention from kernel.py!

        # Feed-forward network
        self.ffn = FeedForward(config)

        # Dropout for residual connections
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, kernel_fn=None) -> torch.Tensor:
        """
        Args:
            x: [B, T, hidden_dim]
            kernel_fn: compiled TileLang kernel (optional, for GPU acceleration)

        Returns:
            [B, T, hidden_dim]
        """
        # --- Attention sub-layer ---
        residual = x
        x = self.ln1(x)

        # Project to Q, K, V
        Q, K, V = self.qkv_proj(x)

        # Run NSA! This is where our three "readers" work.
        nsa_result = self.nsa(Q, K, V, kernel_fn=kernel_fn, is_causal=True)
        attn_output = nsa_result["output"]  # [B, T, HQ, D]

        # Reshape from multi-head back to hidden_dim
        B, T, HQ, D = attn_output.shape
        attn_output = attn_output.reshape(B, T, HQ * D)

        # Output projection + residual
        x = residual + self.dropout(self.qkv_proj.wo(attn_output.float()))

        # --- FFN sub-layer ---
        residual = x
        x = self.ln2(x)
        x = residual + self.ffn(x)

        return x


# ===========================================================================
# Full Language Model
# ===========================================================================
class NSALanguageModel(nn.Module):
    """
    A complete autoregressive language model using NSA.

    This is a "mini-GPT" that can:
        1. Take a sequence of token IDs as input
        2. Output predicted probabilities for the next token at each position
        3. Generate new text autoregressively

    Architecture:
        Token Embedding + Positional Embedding
        -> N x NSATransformerBlock
        -> LayerNorm
        -> Linear Output Head
        -> Logits [B, T, vocab_size]
    """
    def __init__(self, config: NSAModelConfig):
        super().__init__()
        self.config = config

        # --- Embeddings ---
        # Token embedding: each token ID maps to a vector of size hidden_dim
        self.token_emb = nn.Embedding(config.vocab_size, config.hidden_dim)

        # Positional embedding: each position maps to a vector
        # (Absolute positional embeddings for simplicity; RoPE would be better)
        self.pos_emb = nn.Embedding(config.seq_len, config.hidden_dim)

        self.dropout = nn.Dropout(config.dropout)

        # --- Shared NSA module ---
        # We create ONE NativeSparseAttention configuration and share it
        # across all layers. Each layer gets its own QKV projections, but
        # the NSA parameters (compression conv, gates) are per-layer.
        self.blocks = nn.ModuleList()
        for _ in range(config.n_layers):
            nsa = NativeSparseAttention(
                dim=config.head_dim,
                kv_heads=config.n_kv_heads,
                q_heads=config.n_heads,
                block_size=config.block_size,
                selected_blocks=config.selected_blocks,
                compression_ratio=config.compression_ratio,
                window_size=config.window_size,
            )
            self.blocks.append(NSATransformerBlock(config, nsa))

        # --- Output ---
        self.ln_final = nn.LayerNorm(config.hidden_dim)
        self.output_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Weight tying: share weights between token embedding and output head
        # This is a common trick that improves performance and reduces parameters
        self.output_head.weight = self.token_emb.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Count parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"  Model initialized: {n_params / 1e6:.2f}M parameters")

    def _init_weights(self, module):
        """Xavier-style initialization."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,    # [B, T] token IDs
        targets: torch.Tensor = None,  # [B, T] target token IDs (shifted)
        kernel_fn=None,
    ):
        """
        Forward pass.

        Args:
            input_ids: [B, T] integer tensor of token IDs
            targets: [B, T] next-token IDs for computing loss (optional)
            kernel_fn: compiled TileLang kernel (optional)

        Returns:
            logits: [B, T, vocab_size]
            loss: scalar (if targets provided)
        """
        B, T = input_ids.shape
        device = input_ids.device

        # --- Embeddings ---
        tok_emb = self.token_emb(input_ids)                         # [B, T, hidden_dim]
        pos_ids = torch.arange(T, device=device).unsqueeze(0)       # [1, T]
        pos_emb = self.pos_emb(pos_ids)                             # [1, T, hidden_dim]
        x = self.dropout(tok_emb + pos_emb)                         # [B, T, hidden_dim]

        # --- Transformer Blocks ---
        for block in self.blocks:
            x = block(x, kernel_fn=kernel_fn)

        # --- Output Head ---
        x = self.ln_final(x.float())
        logits = self.output_head(x)                                # [B, T, vocab_size]

        # --- Loss ---
        loss = None
        if targets is not None:
            # Cross-entropy loss: compare predicted logits to actual next tokens
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, kernel_fn=None):
        """
        Generate text autoregressively.

        Start with a prompt, predict the next token, append it, repeat.

        Args:
            prompt_ids: [1, T_prompt] starting token IDs
            max_new_tokens: how many new tokens to generate
            temperature: controls randomness (lower = more deterministic)
            kernel_fn: compiled TileLang kernel (optional)
        """
        self.eval()
        generated = prompt_ids.clone()

        for _ in range(max_new_tokens):
            # Crop to max sequence length
            context = generated[:, -self.config.seq_len:]

            # Forward pass
            logits, _ = self.forward(context, kernel_fn=kernel_fn)

            # Get logits for the LAST position only
            logits = logits[:, -1, :] / temperature  # [1, vocab_size]

            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [1, 1]

            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)

        return generated


# ===========================================================================
# Dataset
# ===========================================================================
class CharDataset(Dataset):
    """
    Character-level text dataset.

    Takes raw text, converts each character to its ASCII code (0-255),
    and creates (input, target) pairs where target is shifted by 1 position.

    Example:
        Text:   "Hello"
        Input:  [72, 101, 108, 108]   (H, e, l, l)
        Target: [101, 108, 108, 111]  (e, l, l, o)
    """
    def __init__(self, text: str, seq_len: int):
        self.seq_len = seq_len

        # Convert text to ASCII byte values (0-255)
        self.data = torch.tensor([ord(c) % 256 for c in text], dtype=torch.long)

        print(f"  Dataset: {len(self.data):,} characters, "
              f"{len(self)} sequences of length {seq_len}")

    def __len__(self):
        return max(0, len(self.data) - self.seq_len - 1)

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.seq_len + 1]
        return chunk[:-1], chunk[1:]  # (input, target)


class SyntheticDataset(Dataset):
    """
    A synthetic dataset that generates repeating patterns.

    This is useful for testing without needing a real text file.
    The model should learn to predict the next element in the pattern.

    Pattern: "abcdefghijklmnopqrstuvwxyz0123456789 " repeated
    """
    def __init__(self, seq_len: int, num_sequences: int = 2000):
        self.seq_len = seq_len

        # Create a repeating pattern
        pattern = "abcdefghijklmnopqrstuvwxyz0123456789 " * 1000
        self.data = torch.tensor([ord(c) % 256 for c in pattern[:num_sequences * seq_len + 1]], dtype=torch.long)

        print(f"  Synthetic dataset: {len(self.data):,} tokens, "
              f"{len(self)} sequences of length {seq_len}")

    def __len__(self):
        return max(0, len(self.data) - self.seq_len - 1)

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


# ===========================================================================
# TileLang Kernel Compilation Helper
# ===========================================================================
def compile_kernel(config: NSAModelConfig, device: torch.device):
    """
    Compile the TileLang NSA kernel for the given model configuration.

    Returns None if CUDA is not available or TileLang fails.
    """
    if not TILELANG_AVAILABLE:
        print("  TileLang not available — using PyTorch fallback for selected attention.")
        return None

    if not torch.cuda.is_available():
        print("  No CUDA device — using PyTorch fallback for selected attention.")
        return None

    try:
        print("  Compiling TileLang NSA kernel...")
        groups = config.n_heads // config.n_kv_heads
        kernel_fn = compile_nsa_kernel(
            batch=config.batch_size,
            heads=config.n_heads,
            seq_len=config.seq_len,
            dim=config.head_dim,
            is_causal=True,
            block_size=config.block_size,
            groups=groups,
            selected_blocks=config.selected_blocks,
        )
        print("  TileLang kernel compiled successfully!")
        return kernel_fn
    except Exception as e:
        print(f"  TileLang compilation failed: {e}")
        print("  Falling back to PyTorch attention.")
        return None


# ===========================================================================
# Training Loop
# ===========================================================================
def train(config: NSAModelConfig, data_path: str = None):
    """
    Full training pipeline.

    Steps:
        1. Create dataset and dataloader
        2. Build the NSALanguageModel
        3. Compile TileLang kernel (if available)
        4. Run training loop
        5. Generate sample text
    """
    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  NSA Language Model Training")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # --- Dataset ---
    print("[1/5] Preparing dataset...")
    if data_path and os.path.exists(data_path):
        with open(data_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        dataset = CharDataset(text, config.seq_len)
    else:
        if data_path:
            print(f"  Warning: '{data_path}' not found, using synthetic data.")
        dataset = SyntheticDataset(config.seq_len)

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,  # Important: NSA kernel expects exact batch size
    )

    # --- Model ---
    print("\n[2/5] Building model...")
    model = NSALanguageModel(config).to(device)

    # Use float32 for training stability; the NSA module handles fp16 internally
    # where needed (TileLang kernel operates in fp16)
    dtype = torch.float32

    # --- Kernel ---
    print("\n[3/5] Compiling TileLang kernel...")
    kernel_fn = compile_kernel(config, device)

    # --- Optimizer ---
    print("\n[4/5] Starting training...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),  # Standard LLM betas
    )

    # Cosine learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.max_steps,
        eta_min=config.learning_rate / 10,
    )

    # --- Training Loop ---
    model.train()
    step = 0
    total_loss = 0.0
    start_time = time.time()

    while step < config.max_steps:
        for input_ids, targets in dataloader:
            if step >= config.max_steps:
                break

            input_ids = input_ids.to(device)
            targets = targets.to(device)

            # Forward pass
            logits, loss = model(input_ids, targets=targets, kernel_fn=kernel_fn)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (prevents exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            step += 1

            # --- Logging ---
            if step % config.eval_interval == 0 or step == 1:
                avg_loss = total_loss / min(step, config.eval_interval)
                elapsed = time.time() - start_time
                tokens_per_sec = (step * config.batch_size * config.seq_len) / elapsed
                lr = scheduler.get_last_lr()[0]

                print(f"  Step {step:>5d}/{config.max_steps} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"LR: {lr:.2e} | "
                      f"Tok/s: {tokens_per_sec:.0f} | "
                      f"Elapsed: {elapsed:.1f}s")
                total_loss = 0.0

    total_time = time.time() - start_time
    print(f"\n  Training complete in {total_time:.1f}s")

    # --- Generation ---
    print(f"\n[5/5] Generating sample text...")
    model.eval()

    # Start with a simple prompt
    prompt = "the "
    prompt_ids = torch.tensor([[ord(c) % 256 for c in prompt]], dtype=torch.long, device=device)

    generated_ids = model.generate(
        prompt_ids,
        max_new_tokens=config.gen_max_tokens,
        temperature=config.gen_temperature,
        kernel_fn=kernel_fn,
    )

    generated_text = "".join([chr(t) for t in generated_ids[0].tolist()])
    print(f"\n  Prompt: '{prompt}'")
    print(f"  Generated text:\n  {'-'*40}")
    print(f"  {generated_text}")
    print(f"  {'-'*40}")

    return model


# ===========================================================================
# Entry Point
# ===========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a mini-GPT with DeepSeek NSA")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to a text file for training data. "
                             "If not provided, uses synthetic data.")
    parser.add_argument("--steps", type=int, default=500,
                        help="Number of training steps (default: 500)")
    parser.add_argument("--seq-len", type=int, default=1024,
                        help="Sequence length / context window (default: 1024)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size (default: 4)")
    parser.add_argument("--n-layers", type=int, default=4,
                        help="Number of Transformer layers (default: 4)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate (default: 3e-4)")

    args = parser.parse_args()

    config = NSAModelConfig(
        max_steps=args.steps,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        n_layers=args.n_layers,
        learning_rate=args.lr,
    )

    print("Model Configuration:")
    for k, v in vars(config).items():
        if not k.startswith("_"):
            print(f"  {k}: {v}")

    train(config, data_path=args.data)
