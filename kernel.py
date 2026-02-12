# ruff: noqa
"""
Native Sparse Attention (TileLang) — densely commented version + safety guard

What this kernel does (conceptually):
- For each query token t, for each KV head h, we only attend to S selected KEY blocks.
- Each selected block is a contiguous chunk of BS tokens in the KV sequence.
- We compute attention in a streaming/online way:
    scores = Q[t] @ K[block]^T
    probs  = softmax(scores)   (but done block-by-block with running max/sum for stability)
    out   += probs @ V[block]
  Then normalize by the running denominator.

Key implementation ideas:
- Use shared memory to stage Q/K/V tiles for reuse.
- Use register fragments for accumulators (acc_s, acc_o) and softmax stats.
- Use exp2() instead of exp() for speed; incorporate log2(e) into the scale.
- Compute O in tiles along the hidden dimension (by = i_v), so multiple CTAs cover D.

IMPORTANT FIX ADDED:
- Explicit bounds guard: i_s + BS <= seq_len
  Without it, K/V copies can read past the end of the sequence if indices are bad.

Recommended debugging:
- After calling the kernel, do torch.cuda.synchronize() to surface async failures at the callsite.
"""

import torch
import tilelang
from tilelang import language as T
import tilelang.testing

tilelang.testing.set_random_seed(0)


@tilelang.jit(
    out_idx=[-1],
    pass_configs={
        # Fast math allows faster approx exp/rcp/etc.
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        # Disable some advanced lowering/scheduling passes for simpler generated code.
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def native_sparse_attention(
    batch,
    heads,
    seq_len,
    dim,
    is_causal,
    scale=None,
    block_size=64,
    groups=1,
    selected_blocks=16,
):
    # ---- Softmax scale preparation ----
    # Standard attention scale = 1/sqrt(dim).
    # Here we use exp2() instead of exp(), so we multiply by log2(e) to convert:
    # exp(x) == exp2(x * log2(e))
    if scale is None:
        scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    else:
        scale = scale * 1.44269504  # log2(e)

    # ---- GQA / grouped attention bookkeeping ----
    # Q has "heads" heads
    # K/V have fewer heads (head_kv) and each KV head is shared by "groups" Q heads.
    head_kv = heads // groups

    # ---- Tensor shapes used by TileLang kernel signature ----
    # Q: [B, T, HQ, D]
    # K/V: [B, T, H, D] where H = head_kv
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, head_kv, dim]

    # BlockIndices: for each (b,t,h) a list of S selected key-block IDs (NOT token IDs).
    # Shape: [B, T, H, S]
    block_indices_shape = [batch, seq_len, head_kv, selected_blocks]
    block_indices_dtype = T.int32

    # ---- Data types ----
    dtype = T.float16          # store Q/K/V/O in fp16
    accum_dtype = T.float32    # accumulate scores/output in fp32 for stability

    # ---- Tiling parameters ----
    BS = block_size            # tokens per sparse block
    # Choose channel tile size BK/BV as power-of-2 >= dim but capped at 128.
    # Example: dim=32 -> 32. dim=96 -> 128.
    block_T = min(128, tilelang.math.next_power_of_2(dim))
    BK = BV = block_T

    # Hidden dimension is split into NV tiles of width BV.
    # Here NV = ceil(dim / BV). If dim <= BV then NV=1.
    NV = tilelang.cdiv(dim, block_T)
    NK = tilelang.cdiv(dim, block_T)

    # This particular kernel assumes dim fits into one BK tile for K (NK==1).
    # If dim were larger than BK, you'd need looped K tiles or different layout.
    assert NK == 1, "The key dimension can not be larger than 256"

    # S = number of selected sparse blocks per query token
    S = selected_blocks
    # G = number of Q heads per KV head (groups)
    G = groups

    # Pipeline stages (double-buffer style). Kept small/simple.
    num_stages = 2
    # threads=32 => 1 warp per CTA (simple baseline).
    threads = 32

    @T.prim_func
    def native_sparse_attention(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        BlockIndices: T.Tensor(block_indices_shape, block_indices_dtype),
        Output: T.Tensor(q_shape, dtype),
    ):
        # ------------------------------------------------------------
        # Grid mapping (T.Kernel):
        #   bx : query token index t in [0, seq_len)
        #   by : output-V tile index i_v in [0, NV)
        #   bz : flattened (batch, kv_head) in [0, batch * head_kv)
        #
        # One CTA computes:
        #   - for a single token t
        #   - for a single (batch b, kv head h)
        #   - for a single tile of output channels [i_v*BV : (i_v+1)*BV]
        #   - producing output for G query heads mapped to that KV head.
        # ------------------------------------------------------------
        with T.Kernel(seq_len, NV, batch * head_kv, threads=threads) as (bx, by, bz):

            # -------------------------
            # Shared memory allocations
            # -------------------------
            # Q_shared: [G, BK] query vectors for G heads at token t
            Q_shared = T.alloc_shared([G, BK], dtype)
            # K_shared: [BS, BK] keys for one selected block (BS tokens)
            K_shared = T.alloc_shared([BS, BK], dtype)
            # V_shared: [BS, BV] values for one selected block and one V tile
            V_shared = T.alloc_shared([BS, BV], dtype)
            # O_shared: [G, BV] staging output tile before writing to global
            O_shared = T.alloc_shared([G, BV], dtype)

            # ---------------------------------
            # Register fragments (fast registers)
            # ---------------------------------
            # acc_s: [G, BS] scores or probs for this block (fp32)
            acc_s = T.alloc_fragment([G, BS], accum_dtype)
            # acc_s_cast: [G, BS] probs cast to fp16 for tensorcore GEMM with V
            acc_s_cast = T.alloc_fragment([G, BS], dtype)
            # acc_o: [G, BV] output accumulator for this output tile (fp32)
            acc_o = T.alloc_fragment([G, BV], accum_dtype)

            # Softmax streaming statistics per head (length G):
            scores_max = T.alloc_fragment([G], accum_dtype)       # current max
            scores_max_prev = T.alloc_fragment([G], accum_dtype)  # previous max
            scores_scale = T.alloc_fragment([G], accum_dtype)     # exp2((prev-max)-(cur-max)) factor
            scores_sum = T.alloc_fragment([G], accum_dtype)       # sum of exp2(scores - max)
            logsum = T.alloc_fragment([G], accum_dtype)           # running denominator across blocks

            # -------------------------
            # Decode program indices
            # -------------------------
            i_t = bx    # token index
            i_v = by    # V/output channel tile index
            i_bh = bz   # flattened batch/head

            i_b = i_bh // head_kv  # batch index
            i_h = i_bh % head_kv   # KV head index (0..head_kv-1)

            # -------------------------------------------------------
            # Load Q for this token into shared:
            # Q[b, t, heads, dim]
            #
            # Query heads mapped to this KV head:
            #   heads range = [i_h*G : (i_h+1)*G]
            # Because groups = G queries share one KV head.
            # -------------------------------------------------------
            T.copy(Q[i_b, i_t, i_h * G : (i_h + 1) * G, :], Q_shared)

            # ---------------------------------------
            # Initialize streaming accumulators
            # ---------------------------------------
            T.fill(acc_o, 0)  # output numerator starts at 0
            T.fill(logsum, 0)  # denominator starts at 0
            T.fill(scores_max, -T.infinity(accum_dtype))  # running max starts at -inf

            # -------------------------------------------------------
            # Loop over S selected sparse blocks for this (b,t,h).
            # Pipelined: lets compiler overlap copies and compute.
            # -------------------------------------------------------
            for bi in T.Pipelined(S, num_stages=num_stages):
                # BlockIndices stores block IDs, so multiply by BS to get token start.
                # Example: block id 3 with BS=64 -> token start 192.
                i_s = BlockIndices[i_b, i_t, i_h, bi] * BS

                # -------------------------------
                # SAFETY / VALIDITY GUARD (FIXED)
                # -------------------------------
                # We need:
                #   1) i_s >= 0
                #   2) i_s + BS <= seq_len       (avoid OOB loads)
                #   3) if causal, block must start <= current token
                #
                # NOTE: i_s is token start index; block covers [i_s, i_s+BS).
                if T.all_of(i_s >= 0, i_s + BS <= seq_len, (i_s <= i_t) if is_causal else True):

                    # -----------------------
                    # Load K block into shared
                    # K: [B, T, H, D]
                    # K_shared: [BS, BK]
                    # -----------------------
                    T.copy(K[i_b, i_s : i_s + BS, i_h, :], K_shared)

                    # -------------------------------------------------
                    # Prepare acc_s as mask/bias before GEMM:
                    # We will do: acc_s += Q_shared @ K_shared^T
                    #
                    # If causal:
                    #   for token t, only allow keys <= t.
                    #   keys inside this block correspond to positions (i_s + j).
                    #   if (i_s + j) > t => mask => -inf
                    #
                    # If not causal:
                    #   clear to 0 (no bias)
                    # -------------------------------------------------
                    if is_causal:
                        for gi, j in T.Parallel(G, BS):
                            acc_s[gi, j] = T.if_then_else(
                                i_t >= (i_s + j),  # allowed
                                0,
                                -T.infinity(acc_s.dtype)  # masked
                            )
                    else:
                        T.clear(acc_s)  # set to 0

                    # --------------------------------------
                    # Compute scores for this block:
                    # acc_s += Q_shared @ K_shared^T
                    #
                    # Shapes:
                    #   Q_shared: [G, BK]
                    #   K_shared: [BS, BK] -> transpose_B => [BK, BS]
                    #   result:   [G, BS]
                    # --------------------------------------
                    T.gemm(
                        Q_shared,
                        K_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                    # =========================================================
                    # STREAMING / ONLINE SOFTMAX FOR NUMERICAL STABILITY
                    #
                    # We maintain for each head g:
                    #   m = running max over all processed blocks so far
                    #   l = running sum of exp(score - m) across all blocks so far
                    #
                    # When processing a new block:
                    #   m_new = max(m_prev, max(scores_block))
                    #   scale_prev = exp(m_prev - m_new)
                    #   l_new = l_prev * scale_prev + sum(exp(scores_block - m_new))
                    #
                    # Output accumulator must be rescaled by the same factor:
                    #   O_new = O_prev * scale_prev + (exp(scores_block - m_new) @ V_block)
                    #
                    # Here we use exp2 and "scale" already includes log2(e).
                    # =========================================================

                    # Save previous max
                    T.copy(scores_max, scores_max_prev)

                    # Compute max over this block's scores
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=True)  # per head max across BS

                    # Compute rescale factor for previous accumulators
                    for gi in T.Parallel(G):
                        scores_scale[gi] = T.exp2(
                            scores_max_prev[gi] * scale - scores_max[gi] * scale
                        )

                    # Turn scores -> exp2(scores - max)  (still in acc_s)
                    for gi, j in T.Parallel(G, BS):
                        acc_s[gi, j] = T.exp2(acc_s[gi, j] * scale - scores_max[gi] * scale)

                    # Sum exp2(...) across keys in this block
                    T.reduce_sum(acc_s, scores_sum, dim=1)  # per head sum across BS

                    # Update running denominator: logsum = logsum*scale_prev + block_sum
                    for gi in T.Parallel(G):
                        logsum[gi] = logsum[gi] * scores_scale[gi] + scores_sum[gi]

                    # Cast probabilities to fp16 (for faster GEMM with V)
                    T.copy(acc_s, acc_s_cast)

                    # Rescale previous output accumulator with same factor
                    for gi, j in T.Parallel(G, BV):
                        acc_o[gi, j] *= scores_scale[gi]

                    # --------------------------------------
                    # Load V block tile into shared
                    # V: [B, T, H, D]
                    #
                    # Only load the BV slice for this i_v tile:
                    #   channels [i_v*BV : (i_v+1)*BV]
                    # --------------------------------------
                    T.copy(
                        V[i_b, i_s : i_s + BS, i_h, i_v * BV : (i_v + 1) * BV],
                        V_shared,
                    )

                    # --------------------------------------
                    # Accumulate output:
                    # acc_o += acc_s_cast @ V_shared
                    #
                    # Shapes:
                    #   acc_s_cast: [G, BS]    (probabilities)
                    #   V_shared:   [BS, BV]
                    #   result:     [G, BV]
                    # --------------------------------------
                    T.gemm(
                        acc_s_cast,
                        V_shared,
                        acc_o,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

            # -------------------------------------------------------
            # Final normalization:
            # We accumulated numerator acc_o = sum(exp(score - m) @ V),
            # and logsum = sum(exp(score - m)).
            # So output = acc_o / logsum for each head.
            # -------------------------------------------------------
            for gi, j in T.Parallel(G, BV):
                acc_o[gi, j] /= logsum[gi]

            # Write to global output
            T.copy(acc_o, O_shared)
            T.copy(
                O_shared,
                Output[
                    i_b,
                    i_t,
                    i_h * G : (i_h + 1) * G,
                    i_v * BV : (i_v + 1) * BV,
                ],
            )

    return native_sparse_attention


def main():
    # Test setup
    B, SEQ_LEN, H, HQ, D, S, block_size, dtype, scale = 2, 64, 1, 16, 32, 1, 32, torch.float16, 0.1

    # Compile the kernel with these compile-time constants
    kernel = native_sparse_attention(
        batch=B,
        heads=HQ,
        seq_len=SEQ_LEN,
        dim=D,
        is_causal=True,
        block_size=block_size,
        groups=HQ // H,        # = 16 here (GQA: 16 Q heads share 1 KV head)
        selected_blocks=S,
        scale=scale,
    )

    # Optional: print generated CUDA for inspection
    print(kernel.get_kernel_source())

    # Build random inputs on GPU
    torch.random.manual_seed(0)
    Q = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device="cuda")
    K = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device="cuda")
    V = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device="cuda")

    # Build sparse block indices (block IDs, not token IDs!)
    # Sentinel is SEQ_LEN, which is invalid as a block id.
    # We keep it but the kernel guard will ignore invalid ones safely.
    block_indices = torch.full((B, SEQ_LEN, H, S), SEQ_LEN, dtype=torch.long, device="cuda")

    # Choose blocks only from the past: [0 .. (t//block_size)-1]
    for b in range(B):
        for t in range(SEQ_LEN):
            for h in range(H):
                # number of fully available blocks strictly before token t
                num_avail_blocks = max(1, (t // block_size))
                # pick up to S random blocks
                chosen = torch.randperm(num_avail_blocks, device="cuda")[:S]
                block_indices[b, t, h, : chosen.numel()] = chosen

    # Sort for determinism (optional)
    block_indices = block_indices.sort(-1)[0].to(torch.int32)

    # (Optional) Host-side sanity checks (very useful for avoiding device asserts)
    num_blocks = (SEQ_LEN + block_size - 1) // block_size
    # Allow sentinel SEQ_LEN; filter it out for max/min checks:
    valid = block_indices[block_indices != SEQ_LEN]
    if valid.numel() > 0:
        assert int(valid.min().item()) >= 0
        assert int(valid.max().item()) < num_blocks, (int(valid.max().item()), num_blocks)

    # Run kernel
    out = kernel(Q, K, V, block_indices)

    # Force sync to catch async kernel failures right here
    torch.cuda.synchronize()

    print("out shape:", tuple(out.shape))
    print("out sample:", out[0, 0, 0, :8])


def run_regression_perf():
    B, SEQ_LEN, H, HQ, D, S, block_size, dtype, scale = 2, 64, 1, 16, 32, 1, 32, torch.float16, 0.1

    kernel = native_sparse_attention(
        batch=B,
        heads=HQ,
        seq_len=SEQ_LEN,
        dim=D,
        is_causal=True,
        block_size=block_size,
        groups=HQ // H,
        selected_blocks=S,
        scale=scale,
    )

    torch.random.manual_seed(0)
    Q = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device="cuda")
    K = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device="cuda")
    V = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device="cuda")

    block_indices = torch.full((B, SEQ_LEN, H, S), SEQ_LEN, dtype=torch.long, device="cuda")
    for b in range(B):
        for t in range(SEQ_LEN):
            for h in range(H):
                num_avail_blocks = max(1, (t // block_size))
                chosen = torch.randperm(num_avail_blocks, device="cuda")[:S]
                block_indices[b, t, h, : chosen.numel()] = chosen
    block_indices = block_indices.sort(-1)[0].to(torch.int32)

    from tilelang.profiler import do_bench

    def run_kernel_only():
        kernel(Q, K, V, block_indices)

    return do_bench(run_kernel_only, backend="cupti")


if __name__ == "__main__":
    main()
