# ruff: noqa
# ==================================================================================================
# FULLY COMMENTED (BEGINNER-FRIENDLY) DEMO
# ==================================================================================================
#
# Goal of this script
# -------------------
# This file is a teaching demo for "attention", written for people who are new.
#
# Super short version:
#   - A token is one piece of text (word part).
#   - When producing output for token t, attention decides which older tokens matter.
#   - Dense attention checks almost everything in the past (accurate, expensive).
#   - Sparse attention checks only a few chosen regions (cheaper, still useful).
#
# Think of reading a long book:
#   - Dense attention = re-reading all previous pages every time.
#   - Sparse attention = reading:
#       1) a short summary of old chapters,
#       2) a few important pages,
#       3) the most recent pages.
#
# Technical framing:
#   - Dense causal attention per token is O(T) over history, so total is O(T^2).
#   - Block-sparse selected attention is roughly O(S * block_size) per token.
#   - This file implements a 3-branch NSA design:
#       compressed (global) + selected (sparse) + sliding-window (local).
#
# This file has 2 major jobs:
#   (A) INDEXER (Python / PyTorch):
#       decides WHICH blocks are important.
#
#   (B) KERNEL (TileLang / CUDA):
#       computes attention only on those chosen blocks.
#
# Beginner glossary (used everywhere below):
#   - B: batch size (how many sequences at once)
#   - T: token count (sequence length)
#   - D: head dimension (feature size per head)
#   - H: KV heads (how many key/value head groups)
#   - HQ: query heads (usually >= H when using GQA)
#   - Q/K/V: query/key/value tensors (the 3 standard attention inputs)
#   - Block: contiguous chunk of tokens, size = block_size
#   - S: how many blocks we keep per query token (top-k blocks)
#
# Tensor conventions in this code:
#   - Q shape: [B, T, HQ, D]
#   - K shape: [B, T, H,  D]
#   - V shape: [B, T, H,  D]
#   - groups = HQ // H
#   - BlockIndices shape: [B, T, H, S]
#       stores block IDs (0..num_blocks-1), not raw token indices
#   - BlockCounts shape: [B, T, H]
#       how many entries in BlockIndices are actually valid
#
# Kernel note:
#   If block_id = 7 and block_size = 32, that block starts at token 7*32 = 224.
#   (code: i_s = block_id * block_size)
#
# This demo was previously tested on A100 (SM80).
# ==================================================================================================

import torch
try:
    import tilelang
    from tilelang import language as T
    import tilelang.testing
    TILELANG_AVAILABLE = True
except ImportError:
    TILELANG_AVAILABLE = False
    
    # Dummy mock objects so the code parses when tilelang is unavailable
    class TileLangMock:
        def __getattr__(self, name):
            return TileLangMock()
        def __call__(self, *args, **kwargs):
            return lambda f=None: f if f is not None else TileLangMock()
        def __enter__(self):
            return (0, 0, 0)
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
    
    tilelang = TileLangMock()
    T = TileLangMock()


import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import warnings

# Optional dependency:
# - If FlashAttention is installed, we use its native sliding-window kernel.
# - If not installed, we fall back to a dense PyTorch implementation.
# This mirrors the "fast path + safe fallback" pattern used in many research repos.
try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None


# ---------------------------------------------------------------------------
# BEGINNER PRIMER (read this once if attention is new to you)
# ---------------------------------------------------------------------------
# 1) "Attention score" = how relevant an older token is to the current token.
# 2) Softmax turns raw scores into percentages that sum to 1.
# 3) Output token = weighted mix of value vectors using those percentages.
# 4) Causal mode means "no looking into the future".
# 5) Heads are parallel mini-attention channels (different perspectives).
# ---------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Helper: scaled dot product attention, written explicitly for clarity
# -----------------------------------------------------------------------------
# [DEBUG/VERIFICATION] Naive O(T^2) dense attention for output verification
def dense_attention_output(Qrep, K, V, is_causal=True, scale=None):
    """
    Full dense attention reference.

    Plain-English meaning:
      For every query token, compare it with every allowed key token,
      turn those similarity scores into probabilities, then mix values
      using those probabilities.

    In Native Sparse Attention (and GQA in general), the model has many more Query heads than Key/Value heads. For example, if you have 32 query heads but only 8 KV heads, each KV head is shared by a group of 4 query heads.
    Qrep is the query tensor "down-sampled" to match the number of KV heads so that a simple reference attention calculation can be performed for validation.

    Shapes:
      Qrep: [B, T, H, D]
      K:    [B, T, H, D]
      V:    [B, T, H, D]
      O:    [B, T, H, D] (output)

    Why this function exists:
      - It is easy to trust and debug.
      - It is slow for long sequences (roughly T x T work).
      - We use it as a "teacher/reference" to compare sparse outputs.

    Technical details:
      - Scores: logits[b,t,h,k] = <Qrep[b,t,h,:], K[b,k,h,:]> * scale
      - Causal mask sets logits to -inf when k > t
      - Probabilities: softmax over k dimension
      - Output: O[b,t,h,:] = sum_k P[b,t,h,k] * V[b,k,h,:]
    """
    B, T, H, D = Qrep.shape
    assert K.shape == (B, T, H, D)
    assert V.shape == (B, T, H, D)

    # Default scale is 1/sqrt(D) like standard attention.
    if scale is None:
        scale = 1.0 / (D ** 0.5)

    # logits[b,t,h,k] = dot(Qrep[b,t,h,:], K[b,k,h,:])
    # einsum is a way to do matrix multiplication of Qrep and K
    # "bthd,bkhd->bthk" means:
    #   - b: batch
    #   - t: query time
    #   - h: query head
    #   - d: query dimension
    #   - k: key time
    #   - d: key dimension
    #   - ->: output
    #   - bthk: output batch, query time, query head, key time
    logits = torch.einsum("bthd,bkhd->bthk", Qrep.float(), K.float()) * scale

    # Causal mask: disallow attending to future tokens (k > t)
    if is_causal:
        t_idx = torch.arange(T, device=Qrep.device).view(1, T, 1, 1)
        k_idx = torch.arange(T, device=Qrep.device).view(1, 1, 1, T)
        logits = logits.masked_fill(k_idx > t_idx, float("-inf"))

    # Attention probabilities across k
    P = torch.softmax(logits, dim=-1)  # [B,T,H,T]

    # Weighted sum of values
    O = torch.einsum("bthk,bkhd->bthd", P, V.float())  # [B,T,H,D]
    return O


# ==================================================================================================
# DEEPSEEK NSA: THREE-BRANCH ARCHITECTURE (arXiv:2502.11089)
# ==================================================================================================
#
# DeepSeek NSA uses THREE attention paths at the same time, then blends them:
#
#   O = g_cmp * O_compressed + g_slc * O_selected + g_swa * O_window
#
# Branch 1: COMPRESSED (big picture)
#   - Shrinks K/V sequence (fewer tokens) before attention.
#   - Cheap way to keep global context.
#
# Branch 2: SELECTED (important details)
#   - Picks a few important blocks (top-k).
#   - Runs sparse attention only on those blocks.
#   - This is where your TileLang sparse kernel is used.
#
# Branch 3: WINDOW (recent local context)
#   - Always look at the latest nearby tokens (fixed-size window).
#   - Prevents the model from forgetting local continuity.
#
# Gating (the blender knobs):
#   - g_cmp, g_slc, g_swa are learned weights per token/head.
#   - After sigmoid, each gate is in [0,1].
#   - Larger gate value means "trust this branch more".
#
# Why all three?
#   - Compressed only: can lose details.
#   - Selected only: can miss nearby context.
#   - Window only: cannot see far history.
#   - Together: far + important + nearby.
#
# Technical view:
#   - Compressed branch reduces K/V length from T to about T/c, lowering cost.
#   - Selected branch consumes block IDs (top-k blocks) and runs sparse attention.
#   - Window branch enforces local recency via a fixed causal band.
#   - Gates are learned per token/head and blended multiplicatively per branch.
#
# Reference: https://arxiv.org/abs/2502.11089
#            https://github.com/fla-org/native-sparse-attention
# ==================================================================================================


class CompressedAttention(nn.Module):
    """
    Compressed branch (global summary branch).

    If you only remember one thing:
      This branch creates a shorter "summary timeline" of K/V, then attends to that.

    Why this helps:
      - Fewer tokens to attend over -> less compute.
      - Still keeps broad, long-range context.

    Mental picture:
      original:   [t0 t1 t2 t3 t4 t5 t6 t7 ...]
      compressed: [sum(0..3), sum(4..7), ...]   (if compression_ratio=4)

    Beginner note:
      - Q asks "what do I need?"
      - K says "where should I look?"
      - V says "what information should I copy/mix?"
      - We compress K/V, not Q.

    Technical details:
      - K/V are compressed by depthwise Conv1d with stride = compression_ratio (c).
      - Channel layout is flattened to [B, H*D, T] for Conv1d.
      - Left padding keeps causality (no future leakage).
      - Compressed tensors are expanded for GQA when HQ > H.
      - Compute scales from O(T*T*D) to about O(T*(T/c)*D) for this branch.
    """
    def __init__(self, dim: int, kv_heads: int, compression_ratio: int = 4, kernel_size: Optional[int] = None):
        """
        dim:               D, head dimension (e.g., 64)
        kv_heads:          H, number of KV heads
        compression_ratio: c, stride for the conv (how much to compress)
        kernel_size:       conv kernel width (defaults to compression_ratio)
        """
        super().__init__()
        self.compression_ratio = compression_ratio
        self.kernel_size = kernel_size or compression_ratio

        # Depthwise 1D convolution for K compression:
        #   - Groups = kv_heads * dim so each (head, dim_channel) pair has its own kernel
        #   - This is "depthwise" because groups = in_channels = out_channels
        #   - Kernel learns how to summarize consecutive tokens for each channel
        #
        # We treat the channel dimension as (H * D) by reshaping before conv.
        # Conv1d input shape: [B, C, T] where C = H * D
        self.conv_k = nn.Conv1d(
            in_channels=kv_heads * dim,
            out_channels=kv_heads * dim,
            kernel_size=self.kernel_size,
            stride=compression_ratio,
            padding=0,                        # we'll manually pad for causal
            groups=kv_heads * dim,             # depthwise: one filter per channel
            bias=False,
        )

        # Separate conv for V (V may need different compression than K)
        self.conv_v = nn.Conv1d(
            in_channels=kv_heads * dim,
            out_channels=kv_heads * dim,
            kernel_size=self.kernel_size,
            stride=compression_ratio,
            padding=0,
            groups=kv_heads * dim,
            bias=False,
        )

        # Store dims for reshaping
        self.kv_heads = kv_heads
        self.dim = dim

    def compress(self, X: torch.Tensor) -> torch.Tensor:
        """
        Prepare K or V for causal strided Conv1d.

        Plain-English steps:
          1) Reorder dimensions to Conv1d format.
          2) Cast to float32 so dtype matches Conv1d weights.
          3) Left-pad zeros so outputs never use future tokens.

        Technical shape path:
          [B, T, H, D] -> [B, H*D, T] -> left pad -> Conv1d input.
        """
        B, T_len, H, D = X.shape

        # Step 1: Reshape for Conv1d
        # [B, T, H, D] -> [B, T, H*D] -> [B, H*D, T]
        X_flat = X.reshape(B, T_len, H * D).transpose(1, 2)  # [B, H*D, T]

        # Step 1.5: Cast to float32 to match Conv1d weight dtype
        # Input Q/K/V are often fp16 for GPU efficiency, but nn.Conv1d weights
        # default to float32. PyTorch requires input and weight to share dtype.
        X_flat = X_flat.float()

        # Step 2: Causal padding
        # We pad on the LEFT with (kernel_size - 1) zeros
        # This ensures each output position only depends on past + current tokens
        pad_left = self.kernel_size - 1
        X_padded = F.pad(X_flat, (pad_left, 0))  # pad last dim (T) on the left

        return X_padded  # [B, H*D, T + pad_left]

    def forward(
        self,
        Q: torch.Tensor,          # [B, T, HQ, D]  (full resolution queries)
        K: torch.Tensor,          # [B, Tk, H, D]  (full resolution keys)
        V: torch.Tensor,          # [B, Tk, H, D]  (full resolution values)
        groups: int,              # HQ // H (GQA group count)
        is_causal: bool = True,
        scale: Optional[float] = None,
        q_start_pos: int = 0,     # absolute position of Q[:, 0, ...] in KV timeline
    ) -> torch.Tensor:
        """
        Run compressed branch attention.

        Returns:
          - O_compressed: branch output
          - attn_weights: probabilities over compressed tokens
            (used later by the block selector/indexer)

        Technical returns:
          - O_compressed: [B, Tq, HQ, D]
          - attn_weights: [B, Tq, HQ, Tc]
          - K_c, V_c: [B, Tc, H, D]
        """
        B, T_q, HQ, D = Q.shape
        H = K.shape[2]

        if scale is None:
            scale = 1.0 / (D ** 0.5)

        # -----------------------------------------------------------------------
        # (1) Compress K and V
        # -----------------------------------------------------------------------
        K_padded = self.compress(K)                  # [B, H*D, T + pad]
        K_c = self.conv_k(K_padded)                  # [B, H*D, T_c]
        T_c = K_c.shape[-1]
        K_c = K_c.transpose(1, 2).reshape(B, T_c, H, D)  # [B, T_c, H, D]

        V_padded = self.compress(V)                  # [B, H*D, T + pad]
        V_c = self.conv_v(V_padded)                  # [B, H*D, T_c]
        V_c = V_c.transpose(1, 2).reshape(B, T_c, H, D)  # [B, T_c, H, D]

        # -----------------------------------------------------------------------
        # (2) Expand K_c and V_c for GQA
        # -----------------------------------------------------------------------
        # Q has HQ heads, K_c/V_c have H heads.
        # We need to repeat K_c/V_c so each Q head has a matching KV head.
        #
        # K_c: [B, T_c, H, D] -> repeat along head dim -> [B, T_c, HQ, D]
        if groups > 1:
            K_c_exp = K_c.unsqueeze(3).expand(B, T_c, H, groups, D).reshape(B, T_c, HQ, D)
            V_c_exp = V_c.unsqueeze(3).expand(B, T_c, H, groups, D).reshape(B, T_c, HQ, D)
        else:
            K_c_exp = K_c
            V_c_exp = V_c

        # -----------------------------------------------------------------------
        # (3) Compute attention: Q @ K_c^T -> softmax -> @ V_c
        # -----------------------------------------------------------------------
        # logits[b, t, hq, tc] = dot(Q[b,t,hq,:], K_c[b,tc,hq,:]) * scale
        logits = torch.einsum("bthd,bshd->bths", Q.float(), K_c_exp.float()) * scale

        # -----------------------------------------------------------------------
        # (4) Causal mask for compressed tokens
        # -----------------------------------------------------------------------
        # Each compressed token tc covers original tokens [tc*c .. (tc+1)*c - 1].
        # For causal attention, query at position t can only attend to
        # compressed tokens whose LAST original position <= t.
        # Last original position of compressed token tc = (tc+1)*c - 1
        # So the condition is: (tc+1)*c - 1 <= t  =>  tc <= (t+1)/c - 1
        if is_causal:
            c = self.compression_ratio
            # For query at local index t in the current chunk, absolute position is:
            #   t_abs = q_start_pos + t
            # This is required for KV-cache decoding where Q is only a suffix.
            t_idx = torch.arange(T_q, device=Q.device).view(1, T_q, 1, 1)        # [1,Tq,1,1]
            t_abs = t_idx + q_start_pos
            tc_idx = torch.arange(T_c, device=Q.device).view(1, 1, 1, T_c)       # [1,1,1,T_c]
            # Compressed token tc covers original tokens up to (tc+1)*c - 1
            # Allow if (tc+1)*c - 1 <= t, i.e., tc < (t+1)/c
            causal_mask = ((tc_idx + 1) * c - 1) > t_abs  # True = FUTURE = MASK OUT
            logits = logits.masked_fill(causal_mask, float("-inf"))

        # -----------------------------------------------------------------------
        # (5) Softmax and weighted sum
        # -----------------------------------------------------------------------
        attn_weights = torch.softmax(logits, dim=-1)  # [B, T, HQ, T_c]
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        O_compressed = torch.einsum("bths,bshd->bthd", attn_weights, V_c_exp.float())

        return O_compressed, attn_weights, K_c, V_c  # return attn_weights for indexer use


class SlidingWindowAttention(nn.Module):
    """
    Sliding-window branch (recent-context branch).

    If you only remember one thing:
      Each token looks only at the latest `window_size` tokens behind it.

    Why this exists:
      - Recent words are often the most important for grammar/coherence.
      - Sparse top-k selection can miss nearby tokens; this branch protects against that.

    Speed intuition:
      - Dense attention compares against many old tokens.
      - Window attention compares against only a fixed-size recent band.

    Technical details:
      - For query position t, keys are limited to [t-W+1, ..., t] (causal window).
      - Complexity is about O(T * W * D) for fixed W.
      - Fast path uses flash-attn window kernel when shape/runtime constraints match.
      - Fallback path is dense masked attention for correctness portability.
    """

    def __init__(self, window_size: int = 512):
        """
        window_size: number of past tokens to attend to (including current token)
        """
        super().__init__()
        self.window_size = window_size

    def forward(
        self,
        Q: torch.Tensor,          # [B, T, HQ, D]
        K: torch.Tensor,          # [B, Tk, H, D]
        V: torch.Tensor,          # [B, Tk, H, D]
        groups: int,              # HQ // H
        is_causal: bool = True,
        scale: Optional[float] = None,
        q_start_pos: int = 0,     # absolute position of Q[:, 0, ...] in KV timeline
    ) -> torch.Tensor:
        """
        Returns:
          O_window: [B, T, HQ, D]
        """
        B, T_q, HQ, D = Q.shape
        T_k = K.shape[1]
        H = K.shape[2]
        W = self.window_size

        if scale is None:
            scale = 1.0 / (D ** 0.5)

        # Degenerate window: if W <= 0 there is no valid key to attend to.
        # Returning zeros keeps shapes valid and avoids NaNs.
        if W <= 0:
            return torch.zeros_like(Q, dtype=torch.float32)

        # -----------------------------------------------------------------------
        # FAST PATH: FlashAttention window kernel (real O(T * W) behavior)
        # -----------------------------------------------------------------------
        # Why this matters:
        #   A naive masked implementation builds a full [T x T] matrix first, which is
        #   still quadratic in memory/time. FlashAttention's window mode computes only
        #   the needed local band directly on GPU.
        #
        # FlashAttention expects:
        #   q: [B, T, HQ, D]
        #   k: [B, T, H,  D]
        #   v: [B, T, H,  D]
        # and supports GQA where HQ is a multiple of H.
        if (
            flash_attn_func is not None
            and Q.is_cuda
            and K.is_cuda
            and V.is_cuda
            and is_causal
            and (T_q == T_k)
            and (q_start_pos == 0)
        ):
            try:
                # window_size=(left, right). For causal sliding windows we use:
                #   left = W - 1  (look back W-1 tokens)
                #   right = 0     (no look-ahead)
                O_window = flash_attn_func(
                    Q, K, V,
                    softmax_scale=scale,
                    causal=True,
                    window_size=(W - 1, 0),
                )
                return O_window.float()
            except Exception as ex:
                # We intentionally continue with a mathematically equivalent fallback.
                # This keeps the demo robust even if FlashAttention version/API/GPU
                # capabilities do not match.
                warnings.warn(
                    f"FlashAttention window kernel unavailable at runtime; "
                    f"falling back to dense masked path. Reason: {type(ex).__name__}: {ex}",
                    RuntimeWarning,
                )

        # -----------------------------------------------------------------------
        # FALLBACK PATH: Dense logits + masking (correct, but O(T^2))
        # -----------------------------------------------------------------------
        # We keep this path because:
        #   1) It is easy to understand for beginners.
        #   2) It works on CPU or setups without flash-attn.
        #   3) It is a correctness reference for debugging.
        #
        # (1) Expand K/V for GQA (repeat KV heads so each Q head has a partner)
        if groups > 1:
            K_exp = K.unsqueeze(3).expand(B, T_k, H, groups, D).reshape(B, T_k, HQ, D)
            V_exp = V.unsqueeze(3).expand(B, T_k, H, groups, D).reshape(B, T_k, HQ, D)
        else:
            K_exp = K
            V_exp = V

        # (2) Full logits
        # logits[b, t, hq, k] = dot(Q[b,t,hq,:], K[b,k,hq,:]) * scale
        logits = torch.einsum("bthd,bkhd->bthk", Q.float(), K_exp.float()) * scale

        # (3) Build causal + window mask
        t_idx = torch.arange(T_q, device=Q.device).view(1, T_q, 1, 1)
        t_abs = t_idx + q_start_pos
        k_idx = torch.arange(T_k, device=Q.device).view(1, 1, 1, T_k)
        causal_mask = (k_idx > t_abs) if is_causal else torch.zeros_like(k_idx, dtype=torch.bool)
        window_mask = k_idx < (t_abs - W + 1)
        logits = logits.masked_fill(causal_mask | window_mask, float("-inf"))

        # (4) Softmax + weighted sum
        attn_probs = torch.softmax(logits, dim=-1)
        O_window = torch.einsum("bthk,bkhd->bthd", attn_probs, V_exp.float())
        return O_window


@torch.no_grad()
def build_block_indices_from_compressed(
    attn_weights_compressed: torch.Tensor,  # [B, Tq, HQ, T_c]
    compression_ratio: int,
    block_size: int,
    selected_blocks: int,
    groups: int,
    add_local: int = 2,
    kv_seq_len: Optional[int] = None,  # full KV length (if different from Tq)
    q_start_pos: int = 0,              # absolute position of query index 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Turn compressed-attention weights into block IDs for sparse attention.

    Beginner mental model:
      1) Compressed branch tells us which coarse regions look useful.
      2) We convert those coarse scores into block scores.
      3) We keep top-k block IDs per token/head.

    Outputs:
      - block_indices: chosen block IDs, shape [B, Tq, H, S]
      - block_counts: how many of those slots are valid, shape [B, Tq, H]

    Why `block_counts` matters:
      Early tokens do not have many past blocks yet, so not all S slots are real.

    Technical details:
      - We average attention across grouped Q heads to KV-head level.
      - We map compressed token spans to original-token blocks via exact overlap weights.
      - We update top-k online block-by-block (streaming), avoiding a full
        [B, T, H, num_blocks] materialized score tensor.
      - Causality is enforced at block level using absolute query position.
      - Optional local bonus forces near-recent blocks to survive selection.
    """
    B, T_q, HQ, T_c = attn_weights_compressed.shape
    H = HQ // groups
    BS = block_size
    c = compression_ratio
    if kv_seq_len is None:
        kv_seq_len = T_q
    num_blocks = (kv_seq_len + BS - 1) // BS
    device = attn_weights_compressed.device
    S = max(1, int(selected_blocks))

    # -----------------------------------------------------------------------
    # (1) Merge query heads that share the same KV head (GQA grouping)
    # -----------------------------------------------------------------------
    # [B, Tq, HQ, T_c] -> [B, Tq, H, groups, T_c] -> mean over groups -> [B, Tq, H, T_c]
    attn_per_kv = attn_weights_compressed.view(B, T_q, H, groups, T_c).mean(dim=3)

    # -----------------------------------------------------------------------
    # (2) Build a map from compressed token ranges to original token blocks
    # -----------------------------------------------------------------------
    # For each selected block, we store which compressed tokens overlap it, and
    # by how much. Overlap is normalized by compressed-token length so that each
    # compressed token contributes total mass 1.0 across all overlapped blocks.
    #
    # Example:
    #   If a compressed token spans tokens [8,12) and overlaps block A by 3 tokens
    #   and block B by 1 token, then contributions are 0.75 and 0.25 respectively.
    block_to_tc_indices = [[] for _ in range(num_blocks)]
    block_to_tc_weights = [[] for _ in range(num_blocks)]

    for tc in range(T_c):
        tc_start = tc * c
        tc_end = min(tc_start + c, kv_seq_len)
        if tc_start >= kv_seq_len:
            break

        tc_span = max(1, tc_end - tc_start)
        b0 = tc_start // BS
        b1 = (tc_end - 1) // BS

        for blk in range(b0, b1 + 1):
            blk_start = blk * BS
            blk_end = min(blk_start + BS, kv_seq_len)
            overlap = min(tc_end, blk_end) - max(tc_start, blk_start)
            if overlap > 0:
                block_to_tc_indices[blk].append(tc)
                block_to_tc_weights[blk].append(float(overlap) / float(tc_span))

    # -----------------------------------------------------------------------
    # (3) Keep the best S blocks per token/head (top-k, updated block-by-block)
    # -----------------------------------------------------------------------
    # We maintain the current best S blocks for each [b, t, h], updating one
    # candidate block at a time. This avoids creating full block score tensors.
    #
    # top_vals/top_idx shapes:
    #   [B, T, H, S]
    top_vals = torch.full((B, T_q, H, S), float("-inf"), device=device, dtype=torch.float32)
    top_idx = torch.full((B, T_q, H, S), -1, device=device, dtype=torch.long)

    # t_blk[t] = absolute query position block id.
    # Legal causal blocks for token t are in [0 .. t_blk[t]].
    t_abs = (torch.arange(T_q, device=device) + q_start_pos).view(1, T_q, 1)  # [1,Tq,1]
    t_blk = (t_abs // BS).to(torch.long)                                       # [1,Tq,1]

    # A large local bias strongly encourages selecting recent blocks.
    # This mirrors NSA's "always keep local context" behavior.
    local_bonus = 1e3

    for blk in range(num_blocks):
        tc_list = block_to_tc_indices[blk]

        # Compute block score from compressed attention mass.
        # score_blk shape: [B, T, H]
        if len(tc_list) == 0:
            score_blk = torch.zeros((B, T_q, H), device=device, dtype=torch.float32)
        else:
            idx = torch.tensor(tc_list, device=device, dtype=torch.long)
            w = torch.tensor(block_to_tc_weights[blk], device=device, dtype=torch.float32)
            score_blk = (attn_per_kv.index_select(dim=-1, index=idx) * w.view(1, 1, 1, -1)).sum(dim=-1)

        # Causal legality: query token t cannot pick future blocks.
        legal = (blk <= t_blk)  # [1,T,1]
        score_blk = score_blk.masked_fill(~legal, float("-inf"))

        # Local preference: boost current/nearby recent blocks.
        if add_local and add_local > 0:
            is_local = legal & ((t_blk - blk) < add_local)
            score_blk = torch.where(is_local, score_blk + local_bonus, score_blk)

        # Online top-k update:
        #   candidates = current top-k + this new block
        #   keep best S by score
        cand_vals = torch.cat([top_vals, score_blk.unsqueeze(-1)], dim=-1)  # [B,T,H,S+1]
        cand_idx = torch.cat(
            [
                top_idx,
                torch.full((B, T_q, H, 1), blk, device=device, dtype=torch.long),
            ],
            dim=-1,
        )  # [B,T,H,S+1]

        keep = torch.topk(cand_vals, k=S, dim=-1).indices
        top_vals = torch.gather(cand_vals, dim=-1, index=keep)
        top_idx = torch.gather(cand_idx, dim=-1, index=keep)

    # -----------------------------------------------------------------------
    # (4) Fill block_counts and mark unused slots as -1
    # -----------------------------------------------------------------------
    # Near sequence start, not enough past blocks exist. Instead of pretending
    # all S are valid, we report the real count via block_counts.
    max_valid = min(S, num_blocks)
    block_counts = (t_blk + 1).clamp(min=0, max=max_valid).expand(B, T_q, H).contiguous()  # [B,Tq,H]

    rank = torch.arange(S, device=device).view(1, 1, 1, S)
    valid_rank = rank < block_counts.unsqueeze(-1)  # [B,T,H,S]
    top_idx = torch.where(valid_rank, top_idx, torch.full_like(top_idx, -1))

    # -----------------------------------------------------------------------
    # (5) Sort IDs for stable/debug-friendly output
    # -----------------------------------------------------------------------
    # Sorting makes debug printouts easier to read and keeps outputs stable.
    sentinel = num_blocks + 1
    sortable = torch.where(top_idx >= 0, top_idx, torch.full_like(top_idx, sentinel))
    sortable, _ = torch.sort(sortable, dim=-1)
    block_indices = torch.where(sortable == sentinel, torch.full_like(sortable, -1), sortable)

    return block_indices.to(torch.int32), block_counts.to(torch.int32)


class NativeSparseAttention(nn.Module):
    """
    Full three-branch NSA module.

    What this module does:
      - Runs 3 attention branches in parallel:
        1) compressed (global summary),
        2) selected sparse blocks (important details),
        3) sliding window (recent context).
      - Learns 3 gates and blends branch outputs.

    Architecture diagram:
                                Q, K, V
                               /   |   \\
                              /    |    \\
                      Compressed  Selected  Sliding
                      Attention   Attention  Window
                      (global)   (sparse)   (local)
                         |         |          |
                         O_c       O_s        O_w
                          \\        |         /
                           \\       |        /
                      Gated Combination:
                      O = g_c*O_c + g_s*O_s + g_w*O_w
                           where all three gates are learned directly

    If you are brand new:
      - Think of this as an orchestrator class.
      - It does not invent attention math; it wires branches + indexing + gating.
      - You call one forward() and get final output plus debug tensors.

    Technical details:
      - Branch 1 (compressed) is run first to produce selector signal.
      - Indexer converts compressed attention into sparse block IDs/counts.
      - Branch 2 (selected) uses TileLang kernel when shape assumptions hold,
        otherwise a PyTorch correctness fallback.
      - Branch 3 (window) uses flash-attn when possible, fallback otherwise.
      - Final output is per-token/head gated fusion of all 3 branch outputs.
    """

    def __init__(
        self,
        dim: int,                   # D: head dimension (e.g., 64)
        kv_heads: int,              # H: number of KV heads
        q_heads: int,               # HQ: number of query heads
        block_size: int = 64,       # BS: block size for selected attention
        selected_blocks: int = 16,  # S: how many blocks to select
        compression_ratio: int = 4, # c: how much to compress K/V
        window_size: int = 512,     # W: sliding window size
        add_local: int = 2,         # force-include N local blocks in selected branch
    ):
        super().__init__()

        # Store configuration
        self.dim = dim
        self.kv_heads = kv_heads
        self.q_heads = q_heads
        self.groups = q_heads // kv_heads
        self.block_size = block_size
        self.selected_blocks = selected_blocks
        self.compression_ratio = compression_ratio
        self.window_size = window_size
        self.add_local = add_local

        # -----------------------------------------------------------------------
        # Branch 1: Compressed Attention
        # -----------------------------------------------------------------------
        self.compressed_attn = CompressedAttention(
            dim=dim,
            kv_heads=kv_heads,
            compression_ratio=compression_ratio,
        )

        # -----------------------------------------------------------------------
        # Branch 3: Sliding Window Attention
        # -----------------------------------------------------------------------
        self.window_attn = SlidingWindowAttention(window_size=window_size)

        # -----------------------------------------------------------------------
        # Learned Gating Projections
        # -----------------------------------------------------------------------
        # We learn all three branch gates directly from the query:
        #   g_cmp (compressed), g_slc (selected), g_swa (window).
        #
        # Gate input: Q with shape [B, T, HQ, D]
        # Gate output: [B, T, HQ, 3] -> three logits per head per token
        #
        # Implementation detail:
        # one small linear layer produces 3 gate logits per head.
        #
        # Note: In the DeepSeek paper, gates are conditioned on the query token
        # representation. We use Q directly since we don't have a separate
        # pre-attention representation.
        #
        # Each head gets its own 3-way gate from its own D-dimensional query.
        self.g_proj = nn.Linear(dim, 3, bias=False)    # per-head 3-way gate projection

        # Neutral initialization:
        # logits=0 -> sigmoid(0)=0.5 for g_cmp/g_slc/g_swa.
        nn.init.zeros_(self.g_proj.weight)

    def forward(
        self,
        Q: torch.Tensor,              # [B, T, HQ, D]
        K: torch.Tensor,              # [B, T, H,  D]
        V: torch.Tensor,              # [B, T, H,  D]
        kernel_fn=None,               # compiled TileLang kernel (optional)
        is_causal: bool = True,
        scale: Optional[float] = None,
        cu_seqlens: Optional[torch.Tensor] = None,  # varlen boundaries [N+1] (packed mode)
        kv_cache: Optional[dict] = None,            # decode cache: {"k":..., "v":...}
        use_cache: bool = False,                    # whether to write back cache
        max_cache_len: Optional[int] = None,        # optional sliding cache cap
        q_start_pos: Optional[int] = None,          # absolute query start in KV timeline
    ) -> dict:
        """
        Run the full NSA pipeline and return output plus diagnostics.

        If you are reading this for the first time:
          - Ignore `cu_seqlens` and `kv_cache` at first.
          - The core flow is simply:
              compressed branch -> pick blocks -> selected branch -> window branch -> blend.

        Args:
          Q, K, V:
            Standard attention inputs.
          kernel_fn:
            Optional compiled sparse kernel for speed.
            If omitted/incompatible, safe PyTorch fallback is used.
          cu_seqlens:
            Optional boundaries for packed variable-length mode.
          kv_cache:
            Optional decode cache with previous K/V history.
          q_start_pos:
            Absolute position of first query token in KV timeline.
            Useful in decode mode.

        Returns:
          Dictionary with:
            - final output,
            - per-branch outputs,
            - gate values,
            - selected block metadata,
            - optional updated cache.

        Technical return shapes:
          - output / O_compressed / O_selected / O_window: [B, Tq, HQ, D]
          - g_cmp / g_slc / g_swa: [B, Tq, HQ]
          - block_indices: [B, Tq, H, S]
          - block_counts: [B, Tq, H]
        """
        # -----------------------------------------------------------------------------
        # VARLEN PATH (multiple sequences packed into one long row)
        # -----------------------------------------------------------------------------
        # We process each segment independently so no attention leaks across sequence
        # boundaries. This is simple, explicit, and easy for beginners to inspect.

        # cu_seqlens stands for Cumulative Sequence Lengths. It is a standard way (used by libraries like FlashAttention) to tell the kernel where one sequence ends and the next begins within a packed tensor. If this variable is provided, the model knows it isn't looking at one giant sequence, but rather a "bus" full of several smaller sequences.
        if cu_seqlens is not None:
            if kernel_fn is not None:
                warnings.warn(
                    "kernel_fn is ignored in varlen mode; using PyTorch/Flash fallback paths.",
                    RuntimeWarning,
                )

            if Q.shape[0] != 1 or K.shape[0] != 1 or V.shape[0] != 1:
                raise ValueError("varlen mode expects batch size 1 with packed tokens.")

            # offsets is a list of integers that tells the model where each sequence begins and ends within the packed tensor.
            offsets = cu_seqlens.to("cpu").tolist()
            # offsets = [0, 10, 25, 30] means:
            # - Sequence 1: tokens 0..9 (length 10)
            # - Sequence 2: tokens 10..24 (length 15)
            # - Sequence 3: tokens 25..29 (length 5)
            segments = [(int(offsets[i]), int(offsets[i + 1])) for i in range(len(offsets) - 1)]

            # -----------------------------------------------------------------------
            # 1) Run the full NSA pipeline and return output plus diagnostics.
            # -----------------------------------------------------------------------
            # out_chunks: list of output chunks for each segment
            # cmp_chunks: list of compressed branch output chunks for each segment
            # slc_chunks: list of selected branch output chunks for each segment
            # swa_chunks: list of window branch output chunks for each segment
            # gcmp_chunks: list of compressed branch gate chunks for each segment
            # gslc_chunks: list of selected branch gate chunks for each segment
            # gswa_chunks: list of window branch gate chunks for each segment
            # idx_chunks: list of block indices for each segment
            # cnt_chunks: list of block counts for each segment

            out_chunks = []
            cmp_chunks = []
            slc_chunks = []
            swa_chunks = []
            gcmp_chunks = []
            gslc_chunks = []
            gswa_chunks = []
            idx_chunks = []
            cnt_chunks = []

            for s, e in segments:
                if e <= s:
                    continue
                seg = self.forward(
                    Q[:, s:e],
                    K[:, s:e],
                    V[:, s:e],
                    kernel_fn=None,          # kernels are shape-specialized; fallback for varlen
                    is_causal=is_causal,
                    scale=scale,
                    cu_seqlens=None,
                    kv_cache=None,
                    use_cache=False,
                    max_cache_len=None,
                    q_start_pos=0,
                )
                out_chunks.append(seg["output"])
                cmp_chunks.append(seg["O_compressed"])
                slc_chunks.append(seg["O_selected"])
                swa_chunks.append(seg["O_window"])
                gcmp_chunks.append(seg["g_cmp"])
                gslc_chunks.append(seg["g_slc"])
                gswa_chunks.append(seg["g_swa"])
                idx_chunks.append(seg["block_indices"])
                cnt_chunks.append(seg["block_counts"])

            if not out_chunks:
                B, _, HQ, D = Q.shape
                H = K.shape[2]
                S = self.selected_blocks
                return {
                    "output": torch.zeros((B, 0, HQ, D), device=Q.device, dtype=torch.float32),
                    "O_compressed": torch.zeros((B, 0, HQ, D), device=Q.device, dtype=torch.float32),
                    "O_selected": torch.zeros((B, 0, HQ, D), device=Q.device, dtype=torch.float32),
                    "O_window": torch.zeros((B, 0, HQ, D), device=Q.device, dtype=torch.float32),
                    "g_cmp": torch.zeros((B, 0, HQ), device=Q.device, dtype=torch.float32),
                    "g_slc": torch.zeros((B, 0, HQ), device=Q.device, dtype=torch.float32),
                    "g_swa": torch.zeros((B, 0, HQ), device=Q.device, dtype=torch.float32),
                    "block_indices": torch.zeros((B, 0, H, S), device=Q.device, dtype=torch.int32),
                    "block_counts": torch.zeros((B, 0, H), device=Q.device, dtype=torch.int32),
                }

            return {
                "output": torch.cat(out_chunks, dim=1),
                "O_compressed": torch.cat(cmp_chunks, dim=1),
                "O_selected": torch.cat(slc_chunks, dim=1),
                "O_window": torch.cat(swa_chunks, dim=1),
                "g_cmp": torch.cat(gcmp_chunks, dim=1),
                "g_slc": torch.cat(gslc_chunks, dim=1),
                "g_swa": torch.cat(gswa_chunks, dim=1),
                "block_indices": torch.cat(idx_chunks, dim=1),
                "block_counts": torch.cat(cnt_chunks, dim=1),
            }

        # -----------------------------------------------------------------------------
        # STANDARD / CACHE-AWARE PATH
        # -----------------------------------------------------------------------------
        B, T_q, HQ, D = Q.shape
        H = K.shape[2]
        groups = self.groups

        if scale is None:
            scale = 1.0 / (D ** 0.5)

        # Build K/V context. In decode mode, K/V are "new chunk" and kv_cache has history.
        K_ctx = K
        V_ctx = V
        if kv_cache is not None:
            k_prev = kv_cache.get("k", None)
            v_prev = kv_cache.get("v", None)
            if (k_prev is None) != (v_prev is None):
                raise ValueError("kv_cache must provide both 'k' and 'v' or neither.")

            if k_prev is not None:
                K_ctx = torch.cat([k_prev, K], dim=1)
                V_ctx = torch.cat([v_prev, V], dim=1)

            # Optional cache cap: keep only the newest `max_cache_len` tokens.
            if max_cache_len is not None and max_cache_len > 0 and K_ctx.shape[1] > max_cache_len:
                K_ctx = K_ctx[:, -max_cache_len:, :, :].contiguous()
                V_ctx = V_ctx[:, -max_cache_len:, :, :].contiguous()

            if use_cache:
                kv_cache["k"] = K_ctx.detach()
                kv_cache["v"] = V_ctx.detach()

        # Infer absolute query start if not provided.
        # In decode mode this becomes: start of "new chunk" inside K_ctx timeline.
        if q_start_pos is None:
            q_start = max(0, K_ctx.shape[1] - T_q)
        else:
            q_start = int(q_start_pos)

        # ===================================================================
        # BRANCH 1: Compressed Attention
        # ===================================================================
        O_compressed, attn_weights_c, K_c, V_c = self.compressed_attn(
            Q, K_ctx, V_ctx, groups=groups, is_causal=is_causal, scale=scale, q_start_pos=q_start
        )

        # ===================================================================
        # INDEXER: Derive block indices from compressed attention
        # ===================================================================
        block_indices, block_counts = build_block_indices_from_compressed(
            attn_weights_compressed=attn_weights_c,
            compression_ratio=self.compression_ratio,
            block_size=self.block_size,
            selected_blocks=self.selected_blocks,
            groups=groups,
            add_local=self.add_local,
            kv_seq_len=K_ctx.shape[1],
            q_start_pos=q_start,
        )

        # ===================================================================
        # BRANCH 2: Selected Attention (Sparse)
        # ===================================================================
        # Compiled kernel expects "simple" shape assumptions:
        # - Q and K/V have same length
        # - no decode offset
        # If not true, we use the clear PyTorch fallback path.
        kernel_compatible = (kernel_fn is not None) and (q_start == 0) and (T_q == K_ctx.shape[1])
        if kernel_compatible:
            try:
                O_selected = kernel_fn(Q, K_ctx, V_ctx, block_indices, block_counts)
            except TypeError:
                O_selected = kernel_fn(Q, K_ctx, V_ctx, block_indices)
        else:
            if kernel_fn is not None and not kernel_compatible:
                warnings.warn(
                    "kernel_fn fallback: decode/varlen shapes are not compatible with the "
                    "compiled TileLang kernel assumptions; using PyTorch selected attention.",
                    RuntimeWarning,
                )
            O_selected = self._selected_attention_fallback(
                Q, K_ctx, V_ctx, block_indices, block_counts,
                groups=groups, is_causal=is_causal, scale=scale, q_start_pos=q_start
            )
        O_selected = O_selected.float()

        # ===================================================================
        # BRANCH 3: Sliding Window Attention
        # ===================================================================
        O_window = self.window_attn(
            Q, K_ctx, V_ctx, groups=groups, is_causal=is_causal, scale=scale, q_start_pos=q_start
        )

        # ===================================================================
        # GATING: 3-way learned gate from Q (reference-aligned)
        # ===================================================================
        # g_logits: [B, Tq, HQ, 3]
        g_logits = self.g_proj(Q.float())
        g_all = torch.sigmoid(g_logits)
        g_cmp = g_all[..., 0]
        g_slc = g_all[..., 1]
        O_final = (
            g_cmp.unsqueeze(-1) * O_compressed.float()
            + g_slc.unsqueeze(-1) * O_selected.float()
            + g_swa.unsqueeze(-1) * O_window.float()
        )

        out = {
            "output": O_final,
            "O_compressed": O_compressed,
            "O_selected": O_selected,
            "O_window": O_window,
            "g_cmp": g_cmp,
            "g_slc": g_slc,
            "g_swa": g_swa,
            "block_indices": block_indices,
            "block_counts": block_counts,
        }
        if kv_cache is not None and use_cache:
            out["kv_cache"] = kv_cache
        return out

    def _selected_attention_fallback(
        self,
        Q: torch.Tensor,         # [B, Tq, HQ, D]
        K: torch.Tensor,         # [B, Tk, H,  D]
        V: torch.Tensor,         # [B, Tk, H,  D]
        block_indices: torch.Tensor,  # [B, Tq, H, S]
        block_counts: torch.Tensor,   # [B, Tq, H]
        groups: int,
        is_causal: bool = True,
        scale: Optional[float] = None,
        q_start_pos: int = 0,    # absolute position of Q[:,0] in K timeline
    ) -> torch.Tensor:
        """
        Slow but clear reference implementation for selected sparse attention.

        Why it exists:
          - Easy to read and debug.
          - Works even if TileLang kernel is unavailable.

        What it does per query token:
          1) Read selected block IDs.
          2) Gather K/V tokens from those blocks.
          3) Apply causal mask.
          4) Compute regular attention on the gathered tokens only.

        Technical notes:
          - Operates per batch and per KV head in Python loops (not optimized).
          - Uses block_counts to support variable active blocks per token.
          - Supports decode offsets via q_start_pos.
          - Output shape is [B, Tq, HQ, D].
        """
        B, T_q, HQ, D = Q.shape
        T_k = K.shape[1]
        H = K.shape[2]
        S = block_indices.shape[-1]
        BS = self.block_size

        if scale is None:
            scale = 1.0 / (D ** 0.5)

        # This fallback operates per KV-head for simplicity
        # It is not optimized, only correctness-focused.
        O = torch.zeros_like(Q, dtype=torch.float32)

        for b_idx in range(B):
            for h_idx in range(H):
                # Q heads for this KV head: h_idx*groups .. (h_idx+1)*groups
                q_heads_range = slice(h_idx * groups, (h_idx + 1) * groups)
                Q_h = Q[b_idx, :, q_heads_range, :]  # [Tq, G, D]

                for t in range(T_q):
                    q_t = Q_h[t]  # [G, D]
                    t_abs = q_start_pos + t

                    # Variable-count semantics:
                    # each token/head can use fewer than S blocks (especially near t=0).
                    ns = int(block_counts[b_idx, t, h_idx].item())
                    ns = max(0, min(ns, S))
                    raw_blocks = block_indices[b_idx, t, h_idx, :ns].tolist()

                    # Keep order deterministic and avoid duplicates.
                    seen = set()
                    blocks = []
                    for blk_id in raw_blocks:
                        if blk_id < 0:
                            continue
                        if blk_id in seen:
                            continue
                        seen.add(blk_id)
                        blocks.append(blk_id)

                    # Gather K/V from selected blocks
                    k_tokens = []
                    v_tokens = []
                    positions = []
                    for blk_id in blocks:
                        start = blk_id * BS
                        end = min(start + BS, T_k)
                        if start < 0 or start >= T_k:
                            continue
                        k_tokens.append(K[b_idx, start:end, h_idx, :])
                        v_tokens.append(V[b_idx, start:end, h_idx, :])
                        positions.extend(range(start, end))

                    if not k_tokens:
                        continue

                    k_cat = torch.cat(k_tokens, dim=0)  # [N, D]
                    v_cat = torch.cat(v_tokens, dim=0)  # [N, D]
                    pos = torch.tensor(positions, device=Q.device)

                    # Compute scores: [G, N]
                    scores = torch.matmul(q_t.float(), k_cat.float().t()) * scale

                    # Causal mask: mask out positions > t
                    if is_causal:
                        future_mask = pos > t_abs
                        scores[:, future_mask] = float("-inf")

                    # Softmax and weighted sum
                    probs = torch.softmax(scores, dim=-1)
                    probs = torch.nan_to_num(probs, nan=0.0)
                    out_t = torch.matmul(probs, v_cat.float())  # [G, D]
                    O[b_idx, t, q_heads_range, :] = out_t

        return O


# ==================================================================================================
# 2) ASCII VISUALIZATION HELPER
# ==================================================================================================
#
# This prints a simple text "map" so you can SEE block selection patterns.
#
# Each row is a token t.
# Each column is a block id.
# '#' means that block was selected for that token.
#
# Example output for 4 blocks:
#   "##.#" means blocks 0,1,3 selected.
# ==================================================================================================

def ascii_blockmap(
    block_indices: torch.Tensor,
    BS: int,
    T_show: int = 128,
    start_t: int = 0,       # NEW: where to start visualization
    b: int = 0,
    h: int = 0,
    show_ids: bool = False  # NEW: optionally print actual block IDs
):
    """
    Visualize which key-blocks are selected per token.

    block_indices: [B, T, KV_heads, S]
    BS: block_size
    T_show: how many tokens to display
    start_t: starting token index (IMPORTANT for interesting region)
    b: batch index
    h: kv head index
    show_ids: if True, prints selected block ids per token
    """

    T_total = block_indices.shape[1]
    end_t = min(start_t + T_show, T_total)

    idx = block_indices[b, start_t:end_t, h]  # [T_show, S]

    # Total blocks in full sequence
    num_blocks = (T_total + BS - 1) // BS

    print("\n[ASCII Block Map] '#' means selected block for that token")
    print(
        f"Showing tokens {start_t}..{end_t-1} "
        f"(b={b}, kv_head={h}, total_blocks={num_blocks}, S={idx.shape[-1]})\n"
    )

    for local_t in range(idx.shape[0]):
        global_t = start_t + local_t
        row = ["." for _ in range(num_blocks)]

        selected = idx[local_t].tolist()
        for blk in selected:
            if 0 <= blk < num_blocks:
                row[blk] = "#"

        line = f"{global_t:04d} " + "".join(row)

        if show_ids:
            line += f"   {selected}"

        print(line)



# ==================================================================================================
# 3) TILELANG SPARSE ATTENTION KERNEL (YOUR WORKING KERNEL)
# ==================================================================================================
#
# Beginner view:
#   This is the fast GPU engine for the selected sparse branch.
#   It does the same math as attention, but only on chosen token blocks.
#
# Technical view:
#   For each (batch, kv_head, query_token):
#     O = softmax(Q K^T) V
#   but K/V are visited only for S selected blocks (size BS each), not full history.
#
# Why this is fast:
#   - Less memory traffic (only selected blocks loaded).
#   - Tensor-core GEMMs for block compute.
#   - Streaming softmax avoids storing full score matrix.
#
# Streaming softmax intuition:
#   We scan one block at a time and keep running stats:
#     running max, running exp-sum, running output.
#
# Streaming softmax formula sketch:
#   If old max is m_old and new block max is m_new:
#     rescale = exp(score_scale * (m_old - m_new))
#     denom   = denom * rescale + sum(exp(block_scores - m_new))
#     numer   = numer * rescale + exp(block_scores - m_new) @ V_block
#
# TileLang primitives used:
#   - T.alloc_shared: shared memory tiles for reuse in a thread block
#   - T.alloc_fragment: register fragments for fast accumulation
#   - T.gemm: matrix multiply on tensor cores
#   - T.Pipelined: overlap load/compute stages
# ==================================================================================================

tilelang.testing.set_random_seed(0)

@tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def native_sparse_attention(batch, heads, seq_len, dim, is_causal, scale=None, block_size=64, groups=1, selected_blocks=16):

    # ----------------------------------------------------------------------------------------------
    # Softmax scaling
    # ----------------------------------------------------------------------------------------------
    # Beginner view:
    #   Attention scores are divided by sqrt(dim) so they do not get too large.
    #
    # Technical view:
    #   We evaluate exponentials with exp2 for speed:
    #     exp(x) = exp2(x * log2(e))
    #   so we fold log2(e) into the scale constant once up front.
    # ----------------------------------------------------------------------------------------------
    if scale is None:
        scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    else:
        scale = scale * 1.44269504  # log2(e)

    # Number of KV heads in GQA.
    # Beginner: several query heads can share one KV head.
    head_kv = heads // groups

    # Shapes are compile-time constants for the generated kernel.
    # Technical: this specialization is one reason compiled kernels are fast.
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, head_kv, dim]
    block_indices_shape = [batch, seq_len, head_kv, selected_blocks]
    block_counts_shape = [batch, seq_len, head_kv]

    block_indices_dtype = T.int32
    block_counts_dtype = T.int32
    dtype = T.float16           # input/output in fp16
    accum_dtype = T.float32     # accumulate in fp32 for stability

    BS = block_size             # tokens per block
    BK = BV = min(128, tilelang.math.next_power_of_2(dim))  # tile size on D dimension

    # This kernel variant assumes dim <= 256 (so NK == 1).
    # Technical: larger dim would need a different tiling strategy.
    NK = tilelang.cdiv(dim, BK)
    NV = tilelang.cdiv(dim, BV)
    assert NK == 1, "The key dimension can not be larger than 256"

    S = selected_blocks         # number of blocks per token
    G = groups                  # number of Q heads per KV head
    num_stages = 2              # pipeline stages
    threads = 32                # one warp per "instance"

    @T.prim_func
    def native_sparse_attention(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        BlockIndices: T.Tensor(block_indices_shape, block_indices_dtype),
        BlockCounts: T.Tensor(block_counts_shape, block_counts_dtype),
        Output: T.Tensor(q_shape, dtype),
    ):
        # ------------------------------------------------------------------------------------------
        # Kernel launch geometry
        # ------------------------------------------------------------------------------------------
        # Beginner view:
        #   GPU launches many tiny workers. Each worker handles:
        #     - one query token index,
        #     - one output-dimension tile,
        #     - one (batch, kv_head) pair.
        #
        # Technical mapping:
        #   T.Kernel(seq_len, NV, batch * head_kv, threads=threads)
        #   bx -> token index
        #   by -> output tile index along D
        #   bz -> flattened (batch, kv_head)
        # ------------------------------------------------------------------------------------------
        with T.Kernel(seq_len, NV, batch * head_kv, threads=threads) as (bx, by, bz):

            # --------------------------------------------------------------------------------------
            # Shared memory tiles
            # --------------------------------------------------------------------------------------
            # Beginner: shared memory is a fast scratchpad visible to threads in this block.
            # Technical: Q/K/V tiles are staged here to reduce global-memory traffic.
            # --------------------------------------------------------------------------------------
            Q_shared = T.alloc_shared([G, BK], dtype)      # Q tile for all G query heads
            K_shared = T.alloc_shared([BS, BK], dtype)     # K tile for one selected block
            V_shared = T.alloc_shared([BS, BV], dtype)     # V tile for one selected block

            # --------------------------------------------------------------------------------------
            # Register fragments
            # --------------------------------------------------------------------------------------
            # Beginner: registers are the fastest per-thread storage.
            # Technical:
            #   acc_s      -> score/prob tile [G, BS]
            #   acc_s_cast -> fp16 copy of probs for tensor-core GEMM
            #   acc_o      -> output accumulator [G, BV]
            # --------------------------------------------------------------------------------------
            acc_s = T.alloc_fragment([G, BS], accum_dtype)
            acc_s_cast = T.alloc_fragment([G, BS], dtype)      # fp16 probs for gemm
            acc_o = T.alloc_fragment([G, BV], accum_dtype)

            # --------------------------------------------------------------------------------------
            # Streaming softmax variables per query-head-group (size G)
            # --------------------------------------------------------------------------------------
            # Technical roles:
            #   scores_max      current block max per head-group
            #   scores_max_prev previous running max
            #   scores_scale    rescale factor when max changes
            #   scores_sum      current block exp-sum
            #   logsum          running denominator
            scores_max = T.alloc_fragment([G], accum_dtype)
            scores_max_prev = T.alloc_fragment([G], accum_dtype)
            scores_scale = T.alloc_fragment([G], accum_dtype)
            scores_sum = T.alloc_fragment([G], accum_dtype)
            logsum = T.alloc_fragment([G], accum_dtype)

            # Decode indices
            i_t, i_v, i_bh = bx, by, bz
            i_b, i_h = i_bh // head_kv, i_bh % head_kv

            # --------------------------------------------------------------------------------------
            # Load Q for this token and kv head group
            # --------------------------------------------------------------------------------------
            # Beginner: for this kv head, we load the matching group of query heads.
            # Technical: q-head slice is [i_h * G : (i_h + 1) * G].
            # --------------------------------------------------------------------------------------
            T.copy(Q[i_b, i_t, i_h * G : (i_h + 1) * G, :], Q_shared)

            # Initialize accumulators
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            # --------------------------------------------------------------------------------------
            # Loop over selected blocks (sparse pattern)
            # --------------------------------------------------------------------------------------
            # Beginner flow per selected block:
            #   1) load K block
            #   2) score against Q
            #   3) update streaming softmax stats
            #   4) load V block and accumulate output
            #
            # Technical:
            #   BlockIndices stores block IDs, converted to token start by:
            #     i_s = block_id * BS
            # --------------------------------------------------------------------------------------
            # IMPORTANT UPGRADE #1:
            #   Variable number of selected blocks.
            #
            # BlockCounts[b,t,h] tells us how many entries in BlockIndices[b,t,h,:]
            # are valid for this query token/head. This allows:
            #   - fewer active blocks near sequence start
            #   - future extension to per-token adaptive sparsity
            ns = BlockCounts[i_b, i_t, i_h]

            for si in T.Pipelined(S, num_stages=num_stages):
                if si < ns:
                    i_s = BlockIndices[i_b, i_t, i_h, si] * BS  # block start token index

                    # Basic legality:
                    # - i_s must be in range
                    # - i_s must not be future for causal mode
                    if (i_s >= 0) and ((not is_causal) or (i_s <= i_t)):
                        # IMPORTANT UPGRADE #2:
                        #   Tail/partial-block support.
                        #
                        # Old behavior required (i_s + BS) <= seq_len, which skipped
                        # the final partial block. Here we always load BS rows into
                        # shared memory, but out-of-range rows are filled with zeros.
                        # Those rows are later masked with -inf before softmax.
                        for j, k in T.Parallel(BS, BK):
                            token_idx = i_s + j
                            K_shared[j, k] = T.if_then_else(
                                (token_idx < seq_len) and (k < dim),
                                K[i_b, token_idx, i_h, k],
                                0,
                            )

                        # Mask setup:
                        # Beginner: invalid positions get "very negative" so prob ~= 0.
                        # Technical: valid slots are 0 bias; invalid slots are -inf.
                        if is_causal:
                            for gi, j in T.Parallel(G, BS):
                                token_idx = i_s + j
                                acc_s[gi, j] = T.if_then_else(
                                    (token_idx < seq_len) and (i_t >= token_idx),
                                    0,
                                    -T.infinity(acc_s.dtype),
                                )
                        else:
                            for gi, j in T.Parallel(G, BS):
                                token_idx = i_s + j
                                acc_s[gi, j] = T.if_then_else(
                                    token_idx < seq_len,
                                    0,
                                    -T.infinity(acc_s.dtype),
                                )

                        # Technical score compute:
                        #   acc_s = mask + Q @ K^T
                        T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                        # ------------------------------
                        # Streaming softmax update
                        # ------------------------------
                        # Beginner intuition:
                        #   We never materialize full attention scores.
                        #   We keep a running normalization while scanning blocks.
                        #
                        # Technical sequence:
                        #   1) save previous max
                        #   2) compute new max
                        #   3) rescale old accumulators
                        #   4) exponentiate current block scores
                        #   5) update denominator
                        #   6) update numerator/output
                        # 1) Keep previous max
                        T.copy(scores_max, scores_max_prev)

                        # 2) Find new max for this block
                        T.fill(scores_max, -T.infinity(accum_dtype))
                        T.reduce_max(acc_s, scores_max, dim=1, clear=True)

                        # 3) Compute scale factor to rescale old contributions if max changes
                        for gi in T.Parallel(G):
                            scores_scale[gi] = T.exp2(scores_max_prev[gi] * scale - scores_max[gi] * scale)

                        # 4) Exponentiate scores relative to max (stable)
                        for gi, j in T.Parallel(G, BS):
                            acc_s[gi, j] = T.exp2(acc_s[gi, j] * scale - scores_max[gi] * scale)

                        # 5) Sum exp scores for normalization update
                        T.reduce_sum(acc_s, scores_sum, dim=1)

                        # 6) Update running denominator
                        for gi in T.Parallel(G):
                            logsum[gi] = logsum[gi] * scores_scale[gi] + scores_sum[gi]

                        # Cast probabilities to fp16 for tensor-core V GEMM.
                        T.copy(acc_s, acc_s_cast)

                        # Rescale running output when max changes (normalization consistency).
                        for gi, j in T.Parallel(G, BV):
                            acc_o[gi, j] *= scores_scale[gi]

                        # Load V tile with the same tail-safe logic.
                        for j, vv in T.Parallel(BS, BV):
                            token_idx = i_s + j
                            dim_idx = i_v * BV + vv
                            V_shared[j, vv] = T.if_then_else(
                                (token_idx < seq_len) and (dim_idx < dim),
                                V[i_b, token_idx, i_h, dim_idx],
                                0,
                            )

                        # Accumulate block contribution:
                        #   acc_o += probs @ V_block
                        T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            # Final normalization:
            # Beginner: divide by total probability mass.
            # Technical: acc_o / logsum per head-group.
            for gi, j in T.Parallel(G, BV):
                acc_o[gi, j] /= (logsum[gi] + 1e-6)

            # Store output with tail-safe dimension guard.
            # This matters when BV is a power-of-two tile larger than dim.
            for gi, j in T.Parallel(G, BV):
                dim_idx = i_v * BV + j
                if dim_idx < dim:
                    Output[i_b, i_t, i_h * G + gi, dim_idx] = acc_o[gi, j]

    return native_sparse_attention

def plot_block_selection_heatmap(
    block_indices: torch.Tensor,
    block_size: int,
    b: int = 0,
    h: int = 0,
    start_t: int = 0,
    T_show: int = 256,
    title: str = "Block selection heatmap"
):
    """
    Visualize block selection as a binary heatmap:
      rows = tokens
      cols = blocks
      value=1 if block selected for that token
    """

    # ---- SOURCE OF TRUTH: actual tensor length ----
    # block_indices: [B, T, H_kv, S]
    T = block_indices.shape[1]
    S = block_indices.shape[-1]

    # ---- Clamp requested window to valid [0, T] ----
    start_t = max(0, int(start_t))
    end_t = min(T, start_t + int(T_show))

    # If the window is empty, print something useful and return
    if end_t <= start_t:
        print(
            f"[heatmap] Empty window: start_t={start_t}, end_t={end_t}, "
            f"T={T}. Try smaller start_t or smaller T_show."
        )
        return

    # Slice: [T_window, S]
    idx = block_indices[b, start_t:end_t, h]

    num_blocks = (T + block_size - 1) // block_size

    # Build binary matrix M[t, block] = 1 if selected
    M = torch.zeros((end_t - start_t, num_blocks), device="cpu", dtype=torch.float32)

    # Fill selected blocks.
    # Important: indexer now uses -1 for "unused slot", so we must ignore negatives
    # rather than clamping them to 0 (which would fake-select block 0).
    blk = idx.to("cpu")  # [T_window, S]
    for t in range(blk.shape[0]):
        valid = blk[t][blk[t] >= 0]
        if valid.numel() > 0:
            M[t, valid.tolist()] = 1.0

    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(M, aspect="auto", interpolation="nearest")
    plt.title(f"{title} | b={b}, kv_head={h}, t=[{start_t},{end_t}) | S={S}, blocks={num_blocks}")
    plt.xlabel("Block id")
    plt.ylabel("Token t")
    plt.show()

# ==================================================================================================
# 4) DEMO MAIN
# ==================================================================================================
#
# This demo exercises BOTH paths:
#
#   (A) LEGACY PATH: Your existing MiniDSARouter + TileLang kernel (selected branch only)
#       - Trains the router, picks top-k blocks, runs the sparse kernel
#       - This is what you had before
#
#   (B) NSA PATH: Full three-branch DeepSeek NSA architecture
#       - Compressed attention (global, coarse)
#       - Selected attention (sparse, fine-grained, via TileLang kernel)
#       - Sliding window attention (local)
#       - Gated combination with learned per-head gates
#       - Block indices derived from compressed attention weights (NSA-style indexer)
#
# Both paths produce an output of shape [B, T, HQ, D].
# The NSA path is the one that matches the actual DeepSeek paper.
#
# Diagnostics printed:
#   - Per-branch output stats (mean, std, finite check)
#   - Gate values (g_slc, g_swa, g_cmp) showing branch weighting
#   - Cosine similarity between NSA output and dense attention output
#   - Kernel benchmark timing
# ==================================================================================================

def main():
    # ==================================================================
    # Configuration
    # ==================================================================
    # Using a configuration that creates many blocks so block selection
    # patterns are visible and interesting.
    #
    # T=512, BS=32 => 16 blocks total, S=8 selected => 50% sparsity
    B, SEQ_LEN, H, HQ, D = 2, 512, 1, 16, 32
    block_size = 32
    S = 8                       # selected blocks for sparse kernel
    dtype = torch.float16
    scale = 0.1
    groups = HQ // H            # GQA group count
    compression_ratio = 4       # NSA compression (T -> T/4)
    window_size = 128           # sliding window (small for demo; paper uses 512)

    print("=" * 80)
    print("DeepSeek NSA Demo (Three-Branch Sparse Attention)")
    print("=" * 80)
    print(f"Config: B={B}, T={SEQ_LEN}, H={H}, HQ={HQ}, D={D}")
    print(f"        block_size={block_size}, selected_blocks={S}, groups={groups}")
    print(f"        compression_ratio={compression_ratio}, window_size={window_size}")
    print()
    print("GPU:", torch.cuda.get_device_name(0), "cap:", torch.cuda.get_device_capability(0))

    # ==================================================================
    # Compile TileLang kernel (happens once per configuration)
    # ==================================================================
    print("\n--- Compiling TileLang sparse attention kernel ---")
    if TILELANG_AVAILABLE:
        kernel = native_sparse_attention(
            batch=B,
            heads=HQ,
            seq_len=SEQ_LEN,
            dim=D,
            is_causal=True,
            block_size=block_size,
            groups=groups,
            selected_blocks=S,
            scale=scale,
        )
        print("Kernel compiled successfully.")
    else:
        print("TileLang not available. Skipping compilation.")
        kernel = None

    # ==================================================================
    # Create random inputs
    # ==================================================================
    torch.manual_seed(0)
    Q = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device="cuda")
    K = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device="cuda")
    V = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device="cuda")

    print(f"Q shape: {tuple(Q.shape)}")
    print(f"K shape: {tuple(K.shape)}")
    print(f"V shape: {tuple(V.shape)}")

    # Derive dims from actual tensors (safer than using local variables)
    HQ_actual = Q.shape[2]
    H_actual  = K.shape[2]
    D_actual  = Q.shape[-1]
    groups_actual = HQ_actual // H_actual

    # ==================================================================
    # (B) NSA PATH: Full Three-Branch Architecture
    # ==================================================================
    # This is the DeepSeek NSA architecture from arXiv:2502.11089.
    # Three branches: Compressed + Selected + Sliding Window, blended via gating.
    print("\n" + "=" * 80)
    print("(B) NSA PATH: Three-Branch Architecture (arXiv:2502.11089)")
    print("=" * 80)

    nsa = NativeSparseAttention(
        dim=D_actual,
        kv_heads=H_actual,
        q_heads=HQ_actual,
        block_size=block_size,
        selected_blocks=S,
        compression_ratio=compression_ratio,
        window_size=window_size,
        add_local=2,
    ).to("cuda")

    print(f"NSA module parameters: {sum(p.numel() for p in nsa.parameters()):,}")

    # Run the full NSA pipeline with PyTorch Profiler
    from torch.profiler import profile, record_function, ProfilerActivity
    print("Starting NSA forward pass with PyTorch profiler...")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("NSA_Forward"):
            with torch.no_grad():
                result = nsa(Q, K, V, kernel_fn=kernel, is_causal=True, scale=scale)
                
    # Save Chrome Trace
    prof.export_chrome_trace("nsa_trace.json")
    print(f"Profiler trace saved to 'nsa_trace.json'. Open in chrome://tracing")

    # Extract outputs
    O_nsa       = result["output"]
    O_compressed = result["O_compressed"]
    O_selected   = result["O_selected"]
    O_window     = result["O_window"]
    g_slc        = result["g_slc"]
    g_swa        = result["g_swa"]
    g_cmp        = result["g_cmp"]
    block_indices_nsa = result["block_indices"]
    block_counts_nsa = result["block_counts"]

    # --- Branch output diagnostics ---
    print("\n--- Per-Branch Output Diagnostics ---")
    for name, tensor in [
        ("Compressed (global, coarse)", O_compressed),
        ("Selected   (sparse, fine) ", O_selected),
        ("Window     (local)        ", O_window),
        ("NSA Final  (gated blend)  ", O_nsa),
    ]:
        t = tensor.float()
        print(f"  {name}: shape={tuple(t.shape)}, "
              f"finite={torch.isfinite(t).all().item()}, "
              f"mean={t.mean().item():.4f}, std={t.std().item():.4f}")

    # --- Gate diagnostics ---
    print("\n--- Learned Gate Values (averaged over batch, tokens) ---")
    print(f"  g_slc (selected branch):   mean={g_slc.mean().item():.4f}, "
          f"min={g_slc.min().item():.4f}, max={g_slc.max().item():.4f}")
    print(f"  g_swa (window branch):     mean={g_swa.mean().item():.4f}, "
          f"min={g_swa.min().item():.4f}, max={g_swa.max().item():.4f}")
    print(f"  g_cmp (compressed branch): mean={g_cmp.mean().item():.4f}, "
          f"min={g_cmp.min().item():.4f}, max={g_cmp.max().item():.4f}")
    print(f"  Gate-mass average (not constrained to 1.0): {(g_slc + g_swa + g_cmp).mean().item():.6f}")

    # --- Block selection from NSA indexer ---
    print("\n--- NSA Block Selection (from compressed attention weights) ---")
    b, h = 0, 0
    print("Selected blocks (b=0, kv_head=0) for a few tokens:")
    for t in [0, 1, 15, 31, 32, 33, 63, 127, 255, min(511, SEQ_LEN - 1)]:
        if t < SEQ_LEN:
            n_valid = int(block_counts_nsa[b, t, h].item())
            blk_ids = block_indices_nsa[b, t, h, :n_valid].cpu().tolist()
            print(f"  t={t:>3}  count={n_valid:>2}  blk_ids={blk_ids}")

    # Visualize NSA block selection
    ascii_blockmap(block_indices_nsa, BS=block_size, T_show=128, b=0, h=0)

    # ==================================================================
    # Dense comparison: cosine similarity between NSA and dense attention
    # ==================================================================
    print("\n--- Dense Attention Comparison ---")
    # Compute dense (teacher) output for representative Q heads
    rep_qh = torch.arange(H_actual, device="cuda") * groups_actual
    Qrep = Q[:, :, rep_qh, :]  # [B, T, H, D]
    O_dense = dense_attention_output(Qrep, K, V, is_causal=True, scale=scale)  # [B, T, H, D]

    # Compare: expand O_dense to match HQ heads for fair comparison
    # O_dense is [B, T, H, D]; we only compare the representative heads
    O_nsa_rep = O_nsa[:, :, rep_qh, :]  # [B, T, H, D]

    cos_sim = F.cosine_similarity(
        O_nsa_rep.float().flatten(),
        O_dense.float().flatten(),
        dim=0
    ).item()
    print(f"  Cosine similarity (NSA vs Dense): {cos_sim:.4f}")
    print(f"  (Higher is better; 1.0 = identical. Expect >0.5 for reasonable sparsity)")

    # ==================================================================
    # Benchmark: TileLang sparse kernel timing
    # ==================================================================
    print("\n--- Kernel Benchmark ---")

    # Warmup
    for _ in range(20):
        _ = kernel(Q, K, V, block_indices_nsa, block_counts_nsa)
    torch.cuda.synchronize()

    # Timed run with NVTX markers
    iters = 200
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    # Note: torch.cuda.nvtx markers allow NVIDIA Nsight Systems (nsys) to see the range
    torch.cuda.nvtx.range_push("TileLang_Sparse_Kernel_Loop")
    start_evt.record()
    for _ in range(iters):
        if kernel is not None:
            _ = kernel(Q, K, V, block_indices_nsa, block_counts_nsa)
        else:
            _ = nsa._selected_attention_fallback(Q, K, V, block_indices_nsa, block_counts_nsa, groups, True, scale)
    end_evt.record()
    torch.cuda.nvtx.range_pop()

    torch.cuda.synchronize()
    ms = start_evt.elapsed_time(end_evt)
    print(f"  Sparse Focus Sub-Kernel: {ms/iters:.4f} ms/iter (avg over {iters} iters)")

    # ==================================================================
    # Heatmap visualization
    # ==================================================================
    T_total = block_indices_nsa.shape[1]
    start_t = max(0, T_total - 256)

    plot_block_selection_heatmap(
        block_indices_nsa,
        block_size=block_size,
        b=0, h=0,
        start_t=start_t,
        T_show=min(256, T_total),
        title="NSA: Block selection from compressed attention"
    )

    print("\n" + "=" * 80)
    print("Demo complete. Three-branch NSA architecture exercised successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()
