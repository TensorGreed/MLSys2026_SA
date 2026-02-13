# ruff: noqa
# ==================================================================================================
# FULLY COMMENTED (BEGINNER-FRIENDLY) DEMO
# ==================================================================================================
#
# Goal of this script
# -------------------
# You are building a *block-sparse attention* demo (DeepSeek-like idea):
#
#   Instead of attending to ALL past tokens (dense attention),
#   we pick only a few "important" blocks of tokens (Top-K blocks),
#   and do attention only on those blocks.
#
# This is usually much faster for long sequences because:
#   - Dense attention cost per token grows with sequence length (O(T))
#   - Block sparse attention cost grows with selected blocks (O(S * block_size))
#
# Two big parts:
#   (A) INDEXER (Python / PyTorch):
#         Uses content (Q and K) to decide which blocks matter.
#         Produces BlockIndices[b, t, kv_head, s] = block_id.
#
#   (B) KERNEL (TileLang / CUDA):
#         Uses BlockIndices to run attention only on those blocks.
#
# IMPORTANT conventions in this code:
#   - Q shape: [B, T, HQ, D]        (HQ = number of query heads)
#   - K shape: [B, T, H,  D]        (H  = number of key/value heads, "KV heads")
#   - V shape: [B, T, H,  D]
#   - groups = HQ // H              (Group Query Attention / GQA)
#   - BlockIndices shape: [B, T, H, S]
#       values are BLOCK IDs (0..num_blocks-1), NOT token indices
#
# Kernel detail:
#   The kernel converts block_id -> token start offset by:
#       i_s = block_id * block_size
#
# You ran this successfully on A100 (SM80).
# ==================================================================================================

import torch
import tilelang
from tilelang import language as T
import tilelang.testing
import matplotlib.pyplot as plt

# ==================================================================================================
# 1) CONTENT-BASED TOP-K BLOCK INDEXER  (Option 2)
# ==================================================================================================
#
# What problem does this solve?
# -----------------------------
# The kernel is fast *if you tell it which blocks to attend to*.
# But "which blocks matter?" depends on the content of Q and K.
#
# A "DeepSeek-like" idea is:
#   - Create a small "summary key" for each block (one vector per block)
#   - For each token, compare its query to each block summary
#   - Pick top-K blocks by similarity score
#
# This gives you a sparse pattern that is *content-driven* rather than random.
#
# Output:
#   BlockIndices[b, t, kv_head, s]  =  block_id
# where block_id is an integer in [0..num_blocks-1].
#
# Causality:
#   For token t, we only allow blocks whose start <= t (causal attention).
#
# add_local:
#   Many real sparse attention systems mix:
#       - "local blocks" (most recent blocks)
#       - "retrieved blocks" (content-chosen)
#   add_local=2 forces last 2 blocks to always be included (when they exist).
# ==================================================================================================

@torch.no_grad()
def build_block_indices_topk(
    Q: torch.Tensor,         # [B, T, HQ, D]
    K: torch.Tensor,         # [B, T, H,  D]
    block_size: int,
    selected_blocks: int,
    groups: int,
    summary: str = "mean",   # "mean" | "last" | "first"
    add_local: int = 2,      # always include most recent N blocks
) -> torch.Tensor:
    """
    Returns:
        block_indices: int32 tensor of shape [B, T, H, S]
            Each entry is a BLOCK ID (0..num_blocks-1).
    """
    # ---- Basic validation ----
    assert Q.is_cuda and K.is_cuda, "This demo expects CUDA tensors"
    B, T, HQ, D = Q.shape
    Bk, Tk, H, Dk = K.shape
    assert (B, T, D) == (Bk, Tk, Dk), "Q and K must match in B,T,D"
    assert HQ % groups == 0, "HQ must be divisible by groups (GQA requirement)"

    # In GQA, H (KV heads) = HQ / groups
    head_kv = HQ // groups
    assert head_kv == H, f"Expected H={H} == HQ/groups={head_kv}"

    BS = int(block_size)
    S = int(selected_blocks)

    # How many blocks are in the sequence?
    # Example: T=512, BS=32 => num_blocks = 16
    num_blocks = (T + BS - 1) // BS

    device = Q.device

    # ----------------------------------------------------------------------------------------------
    # Step A: Build block summaries of K
    # ----------------------------------------------------------------------------------------------
    # We want: Ksum[b, block_id, kv_head, d]
    #
    # How:
    #   1) Pad K so its length is divisible by block_size
    #   2) Reshape into blocks: [B, num_blocks, BS, H, D]
    #   3) Reduce BS dimension to get one vector per block
    # ----------------------------------------------------------------------------------------------

    pad = num_blocks * BS - T
    if pad:
        # Pad with zeros; those padded tokens should never be used because causal mask prevents it
        Kp = torch.cat(
            [K, torch.zeros((B, pad, H, D), device=device, dtype=K.dtype)],
            dim=1,
        )
    else:
        Kp = K

    # Now reshape into blocks
    # Kb: [B, num_blocks, BS, H, D]
    Kb = Kp.view(B, num_blocks, BS, H, D)

    # Choose how to summarize block into one vector
    if summary == "mean":
        # Mean key vector for the block
        Ksum = Kb.mean(dim=2)              # [B, num_blocks, H, D]
    elif summary == "last":
        # Only take the last token's key in each block
        Ksum = Kb[:, :, BS - 1, :, :]      # [B, num_blocks, H, D]
    elif summary == "first":
        # Only take the first token's key in each block
        Ksum = Kb[:, :, 0, :, :]           # [B, num_blocks, H, D]
    else:
        raise ValueError(f"summary must be one of mean/last/first, got {summary}")

    # ----------------------------------------------------------------------------------------------
    # Step B: Pick a representative Q head per KV head (GQA mapping)
    # ----------------------------------------------------------------------------------------------
    # Q has HQ heads, but K/V have only H heads.
    # groups = HQ/H tells how many query heads share one KV head.
    #
    # Example: HQ=16, H=1, groups=16:
    #   All 16 Q heads share the same K/V head.
    #
    # Here we pick the FIRST q-head in each group as representative for scoring.
    # rep_qh[h] = h*groups
    #
    # Qrep: [B, T, H, D]
    # ----------------------------------------------------------------------------------------------

    rep_qh = torch.arange(H, device=device) * groups  # [H]
    Qrep = Q[:, :, rep_qh, :]                         # [B, T, H, D]

    # ----------------------------------------------------------------------------------------------
    # Step C: Compute relevance score: dot(Qrep, Ksum)
    # ----------------------------------------------------------------------------------------------
    # scores[b, t, h, block] = dot( Qrep[b,t,h,:], Ksum[b,block,h,:] )
    #
    # This is a cheap approximate "which blocks seem relevant?"
    # It's NOT the full attention score (which is token-by-token).
    # It's a retrieval step (coarse).
    #
    # Shape: [B, T, H, num_blocks]
    # ----------------------------------------------------------------------------------------------

    scores = torch.einsum("bthd,bnhd->bthn", Qrep.float(), Ksum.float())

    # ----------------------------------------------------------------------------------------------
    # Step D: Apply causal mask
    # ----------------------------------------------------------------------------------------------
    # For token t, allowed blocks are those whose start token <= t.
    # start token = block_id * BS
    # equivalent condition: block_id <= t//BS
    #
    # We'll mask scores for block_id > t_blk to -inf so they never get selected.
    # ----------------------------------------------------------------------------------------------

    t_blk = (torch.arange(T, device=device) // BS).view(1, T, 1, 1)              # [1,T,1,1]
    blk_ids = torch.arange(num_blocks, device=device).view(1, 1, 1, num_blocks)  # [1,1,1,num_blocks]
    scores = scores.masked_fill(blk_ids > t_blk, float("-inf"))

    # ----------------------------------------------------------------------------------------------
    # Step E: Force include last add_local blocks (local attention)
    # ----------------------------------------------------------------------------------------------
    # This is common in real systems:
    #   Always include recent blocks even if Top-K doesn't pick them.
    #
    # We "boost" their score by a huge constant so they appear in topk.
    # ----------------------------------------------------------------------------------------------

    if add_local and add_local > 0:
        for j in range(add_local):
            local_id = t_blk - j  # most recent block is t_blk, then t_blk-1, etc.
            valid = (local_id >= 0)
            idx = local_id.clamp(min=0).expand(B, T, H, 1)   # shape [B,T,H,1]
            boost = valid.expand(B, T, H, 1).float() * 1e9    # big boost
            scores.scatter_add_(dim=-1, index=idx, src=boost)

    # ----------------------------------------------------------------------------------------------
    # Step F: Top-K selection
    # ----------------------------------------------------------------------------------------------
    # For each (b,t,h), choose K block IDs with highest scores.
    # If there are fewer blocks than S, we pick as many as exist.
    # ----------------------------------------------------------------------------------------------

    k = min(S, num_blocks)
    top = torch.topk(scores, k=k, dim=-1).indices   # [B,T,H,k]
    top, _ = torch.sort(top, dim=-1)                # sorted ascending for nicer printing

    return top.to(torch.int32)


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
# What is this kernel doing conceptually?
# --------------------------------------
# We compute attention output for each token and head:
#
#   Attention(Q, K, V) = softmax(Q*K^T) * V
#
# Dense attention would use ALL past tokens in K and V.
#
# Here we do BLOCK SPARSE attention:
#   - We only consider S selected blocks of size BS
#   - For each selected block:
#       1) load K block and V block into shared memory
#       2) compute scores for that block: Q_block * K_block^T
#       3) do a "streaming softmax" update across blocks
#       4) multiply probabilities by V block and accumulate output
#
# Why streaming softmax?
# ----------------------
# Softmax normally needs all scores at once to compute:
#   exp(score - max) / sum(exp(score - max))
#
# If we process blocks one by one, we can't store all scores.
# So we maintain running statistics:
#   - running max (scores_max)
#   - running sum of exp(score - max) (logsum)
#
# That allows us to process block-by-block without storing the entire attention matrix.
#
# TileLang concepts used:
# -----------------------
# - T.alloc_shared: shared memory tile (fast, per-block)
# - T.alloc_fragment: registers (very fast, per-thread)
# - T.gemm: uses tensor cores / MMA under the hood
# - T.Pipelined: software pipelining of loads/compute
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
    # Standard attention uses:
    #   score = (Q dot K) * (1/sqrt(dim))
    #
    # This kernel uses exp2() instead of exp() for speed:
    #   exp(x) = exp2(x * log2(e))
    #
    # So we fold log2(e) into the scale.
    # ----------------------------------------------------------------------------------------------
    if scale is None:
        scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    else:
        scale = scale * 1.44269504  # log2(e)

    # Number of KV heads in GQA
    head_kv = heads // groups

    # Shapes are compile-time constants for the kernel
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, head_kv, dim]
    block_indices_shape = [batch, seq_len, head_kv, selected_blocks]

    block_indices_dtype = T.int32
    dtype = T.float16           # input/output in fp16
    accum_dtype = T.float32     # accumulate in fp32 for stability

    BS = block_size             # tokens per block
    BK = BV = min(128, tilelang.math.next_power_of_2(dim))  # tile size on D dimension

    # This kernel assumes dim <= 256 (in this configuration NK==1)
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
        Output: T.Tensor(q_shape, dtype),
    ):
        # ------------------------------------------------------------------------------------------
        # Kernel launch geometry
        # ------------------------------------------------------------------------------------------
        # T.Kernel(seq_len, NV, batch * head_kv, threads=threads)
        #
        # Means our 3D grid is:
        #   bx in [0 .. seq_len-1]              token index
        #   by in [0 .. NV-1]                   tile index along D for output
        #   bz in [0 .. batch*head_kv - 1]      combined batch/head index
        #
        # One warp handles:
        #   one token position (bx)
        #   one output tile along D (by)
        #   one (batch, kv_head) (bz)
        # ------------------------------------------------------------------------------------------
        with T.Kernel(seq_len, NV, batch * head_kv, threads=threads) as (bx, by, bz):

            # --------------------------------------------------------------------------------------
            # Shared memory tiles
            # --------------------------------------------------------------------------------------
            # Shared memory is much faster than global memory.
            # We load the active Q/K/V tiles here to reuse them during compute.
            # --------------------------------------------------------------------------------------
            Q_shared = T.alloc_shared([G, BK], dtype)      # Q tile for all G query heads
            K_shared = T.alloc_shared([BS, BK], dtype)     # K tile for one selected block
            V_shared = T.alloc_shared([BS, BV], dtype)     # V tile for one selected block
            O_shared = T.alloc_shared([G, BV], dtype)      # intermediate output tile

            # --------------------------------------------------------------------------------------
            # Register fragments
            # --------------------------------------------------------------------------------------
            # Registers are fastest storage. T.alloc_fragment maps to registers.
            #
            # acc_s: attention scores / probabilities for [G, BS]
            # acc_o: output accumulator for [G, BV]
            # --------------------------------------------------------------------------------------
            acc_s = T.alloc_fragment([G, BS], accum_dtype)
            acc_s_cast = T.alloc_fragment([G, BS], dtype)      # fp16 probs for gemm
            acc_o = T.alloc_fragment([G, BV], accum_dtype)

            # --------------------------------------------------------------------------------------
            # Streaming softmax variables per query-head-group (size G)
            # --------------------------------------------------------------------------------------
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
            # Q has HQ heads; each kv head corresponds to G query heads:
            #   q heads range: [i_h*G : (i_h+1)*G]
            # --------------------------------------------------------------------------------------
            T.copy(Q[i_b, i_t, i_h * G : (i_h + 1) * G, :], Q_shared)

            # Initialize accumulators
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            # --------------------------------------------------------------------------------------
            # Loop over selected blocks (sparse pattern)
            # --------------------------------------------------------------------------------------
            # BlockIndices gives BLOCK IDs.
            # Convert block_id -> start token index by multiplying with BS.
            #
            # We process block by block:
            #   - load K block
            #   - compute scores
            #   - update softmax state
            #   - load V block
            #   - accumulate output
            # --------------------------------------------------------------------------------------
            for si in T.Pipelined(S, num_stages=num_stages):

                i_s = BlockIndices[i_b, i_t, i_h, si] * BS  # block start token index

                # Guard:
                # - causal: start must be <= current token
                # - in-bounds: start+BS must not exceed seq_len
                if (i_s <= i_t) and (i_s >= 0) and ((i_s + BS) <= seq_len):

                    # Load K tile for this block into shared memory
                    T.copy(K[i_b, i_s : i_s + BS, i_h, :], K_shared)

                    # Apply causal masking inside the block if needed
                    if is_causal:
                        for gi, j in T.Parallel(G, BS):
                            acc_s[gi, j] = T.if_then_else(
                                i_t >= (i_s + j),
                                0,
                                -T.infinity(acc_s.dtype),
                            )
                    else:
                        T.clear(acc_s)

                    # Compute scores: acc_s += Q_shared @ K_shared^T
                    T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                    # ------------------------------
                    # Streaming softmax update
                    # ------------------------------
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

                    # Cast probabilities to fp16 for the V multiply GEMM
                    T.copy(acc_s, acc_s_cast)

                    # Rescale output accumulator because denominator/max changed
                    for gi, j in T.Parallel(G, BV):
                        acc_o[gi, j] *= scores_scale[gi]

                    # Load V tile and do: acc_o += probs @ V
                    T.copy(V[i_b, i_s : i_s + BS, i_h, i_v * BV : (i_v + 1) * BV], V_shared)
                    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            # Final normalization: divide by running denominator
            for gi, j in T.Parallel(G, BV):
                acc_o[gi, j] /= (logsum[gi] + 1e-6)

            # Store output
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[i_b, i_t, i_h * G : (i_h + 1) * G, i_v * BV : (i_v + 1) * BV])

    return native_sparse_attention

def plot_block_selection_heatmap(
    block_indices: torch.Tensor,  # [B, T, H, S] int32
    block_size: int,
    b: int = 0,
    h: int = 0,
    start_t: int = 0,
    T_show: int = 256,
    title: str = "Block selection heatmap (token × block)",
):
    """
    Visualize selection as a binary heatmap:
      y-axis: token index t
      x-axis: block id
      value: 1 if block selected for token, else 0

    Works great for showing late-sequence behavior:
      start_t = T - 256, T_show = 256
    """
    assert block_indices.ndim == 4, "Expected [B, T, H, S]"
    B, T, H, S = block_indices.shape
    end_t = min(start_t + T_show, T)
    idx = block_indices[b, start_t:end_t, h]  # [T_show, S]

    num_blocks = (T + block_size - 1) // block_size

    # Build binary matrix M[t, block] = 1 if selected
    M = torch.zeros((end_t - start_t, num_blocks), device="cpu", dtype=torch.float32)

    # Fill selected blocks
    # (We clamp to valid range just in case)
    blk = idx.to("cpu").clamp(min=0, max=num_blocks - 1)  # [T_show, S]
    t_ids = torch.arange(end_t - start_t).unsqueeze(1).repeat(1, S)  # [T_show, S]
    M[t_ids, blk] = 1.0

    plt.figure(figsize=(12, 6))
    plt.imshow(M.numpy(), aspect="auto", interpolation="nearest")  # default colormap
    plt.xlabel("Block ID")
    plt.ylabel("Token t (windowed)")
    plt.title(f"{title} | b={b}, kv_head={h}, t=[{start_t}..{end_t-1}], blocks={num_blocks}, S={S}")
    plt.colorbar(label="selected (1) / not selected (0)")
    plt.show()

# ==================================================================================================
# 4) DEMO MAIN
# ==================================================================================================
#
# What this does:
#   - sets a demo configuration that creates many blocks
#   - compiles the TileLang kernel
#   - creates random Q/K/V
#   - builds block indices with content-based Top-K
#   - prints selected block IDs for some tokens
#   - prints ASCII map of block selection
#   - warms up and benchmarks kernel with CUDA events
#   - prints output statistics and a sample
# ==================================================================================================

def main():
    # Use a configuration where block selection becomes visible:
    # T=512, BS=32 => 16 blocks total
    B, SEQ_LEN, H, HQ, D = 2, 512, 1, 16, 32
    block_size = 32
    S = 8
    dtype = torch.float16
    scale = 0.1
    groups = HQ // H

    print("GPU:", torch.cuda.get_device_name(0), "cap:", torch.cuda.get_device_capability(0))

    # Compile kernel (happens once per configuration)
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

    # Create inputs
    torch.manual_seed(0)
    Q = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device="cuda")
    K = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device="cuda")
    V = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device="cuda")

    # Build content-based Top-K block indices
    block_indices = build_block_indices_topk(
        Q=Q,
        K=K,
        block_size=block_size,
        selected_blocks=S,
        groups=groups,
        summary="mean",
        add_local=2,  # always include last 2 blocks
    )

    # Show selected blocks for a few tokens
    b, h = 0, 0
    print("\nSelected blocks (b=0, kv_head=0) for a few tokens:")
    for t in [0, 1, 15, 31, 32, 33, 63, 127, 255, 511]:
        print(f"t={t:>3}  blk_ids={block_indices[b, t, h].cpu().tolist()}")

    # Visualize selection for first 128 tokens
    ascii_blockmap(block_indices, BS=block_size, T_show=128, b=0, h=0)

    plot_block_selection_heatmap(
        block_indices,
        block_size=block_size,
        b=0, h=0,
        start_t=SEQ_LEN - 256,
        T_show=256,
    )

    # Warmup runs (helps stabilize performance measurement)
    for _ in range(20):
        out = kernel(Q, K, V, block_indices)
    torch.cuda.synchronize()

    # Benchmark using CUDA events (accurate GPU timing)
    iters = 200
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        out = kernel(Q, K, V, block_indices)
    end.record()

    torch.cuda.synchronize()
    ms = start.elapsed_time(end)
    print(f"\nKernel time: {ms/iters:.4f} ms/iter  (avg over {iters} iters)")

    # Output sanity checks
    print("out shape:", tuple(out.shape))
    print("finite:", torch.isfinite(out).all().item())
    print("mean/std:", out.float().mean().item(), out.float().std().item())

    # Print a small slice of output (demo output)
    print("sample:", out[0, 0, 0, :8].float().cpu().tolist())


if __name__ == "__main__":
    main()
