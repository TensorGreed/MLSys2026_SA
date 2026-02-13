# ruff: noqa
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
# These two often matter for CUTLASS/CUTE code paths:
os.environ["CUDAARCHS"] = "80"
os.environ["TVM_CUDA_ARCH"] = "sm_80"   # if TVM honors it in this build

import torch
import tilelang
from tilelang import language as T
import tilelang.testing

# ------------------------------------------------------------
# Seeding: tilelang.testing.set_random_seed sets Python, NumPy,
# Torch CPU, and Torch CUDA seeds (if CUDA is available).
# NOTE: if you previously hit a device-side assert, the CUDA
# context can be "poisoned" and even seeding can error.
# In that case: restart the notebook kernel/runtime.
# ------------------------------------------------------------
tilelang.testing.set_random_seed(0)


@tilelang.jit(
    # out_idx=[-1] means: the last argument of the prim_func is the output buffer
    out_idx=[-1],
    pass_configs={
        # Fast math allows faster approximate math (exp2, etc.)
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        # Disable some advanced lowering/scheduling features for stability/simplicity
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
    # ------------------------------------------------------------
    # Softmax scaling
    # ------------------------------------------------------------
    # Normally attention uses exp(x). Here code uses exp2(x).
    # exp(x) = exp2(x * log2(e)) so we multiply scale by log2(e).
    if scale is None:
        scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    else:
        scale = scale * 1.44269504  # log2(e)

    # ------------------------------------------------------------
    # Grouped Query Attention (GQA) bookkeeping:
    # - Q has `heads` heads
    # - K/V have `head_kv = heads // groups` heads
    # - Each KV head serves `groups` query heads
    # ------------------------------------------------------------
    head_kv = heads // groups

    # Q and Output shapes: [B, T, HQ, D]
    q_shape = [batch, seq_len, heads, dim]
    # K and V shapes: [B, T, H, D] where H = head_kv
    kv_shape = [batch, seq_len, head_kv, dim]

    # Sparse block indices:
    # BlockIndices[b, t, h, i] gives i-th selected BLOCK-ID for token t and KV head h.
    # Kernel multiplies this block-id by BS to get token start i_s.
    block_indices_shape = [batch, seq_len, head_kv, selected_blocks]
    block_indices_dtype = T.int32

    # Compute/storage dtypes
    dtype = T.float16
    accum_dtype = T.float32

    # ------------------------------------------------------------
    # Tiling sizes
    # ------------------------------------------------------------
    # BS: number of tokens per block
    block_S = block_size
    # block_T: tile size along hidden dimension (power-of-2, capped at 128)
    block_T = min(128, tilelang.math.next_power_of_2(dim))

    # NK/NV: number of tiles along D for K and V (ceil division)
    NK = tilelang.cdiv(dim, block_T)
    NV = tilelang.cdiv(dim, block_T)

    # This implementation assumes dim fits in one tile (NK==1).
    assert NK == 1, "The key dimension can not be larger than 256"

    # Aliases
    S = selected_blocks   # number of selected blocks per query token
    G = groups            # number of Q heads per KV head
    BS = block_S          # tokens per selected block
    BK = BV = block_T     # channels per tile for K and V/O

    # Pipelining settings
    num_stages = 2 if selected_blocks >= 2 else 1
    # 32 threads = one warp per CTA (simple baseline)
    threads = 32

    # ------------------------------------------------------------
    # The actual GPU kernel (TIR / prim_func)
    # ------------------------------------------------------------
    @T.prim_func
    def native_sparse_attention(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        BlockIndices: T.Tensor(block_indices_shape, block_indices_dtype),
        Output: T.Tensor(q_shape, dtype),
    ):
        # --------------------------------------------------------
        # Kernel launch grid:
        #   bx in [0..seq_len)          -> token index i_t
        #   by in [0..NV)               -> V/O tile index i_v
        #   bz in [0..batch*head_kv)    -> combined (batch, kv_head)
        # --------------------------------------------------------
        with T.Kernel(seq_len, NV, batch * head_kv, threads=threads) as (bx, by, bz):

            # -------------------------
            # Shared memory buffers
            # -------------------------
            # Q_shared holds G query heads for this token (one token t)
            Q_shared = T.alloc_shared([G, BK], dtype)
            # K_shared holds one selected K block (BS tokens)
            K_shared = T.alloc_shared([BS, BK], dtype)
            # V_shared holds one selected V block (BS tokens), but only BV channels of this i_v tile
            V_shared = T.alloc_shared([BS, BV], dtype)
            # O_shared is a staging area before writing output
            O_shared = T.alloc_shared([G, BV], dtype)

            # -------------------------
            # Register fragments
            # -------------------------
            # acc_s: [G, BS] - attention scores/probs for this block
            acc_s = T.alloc_fragment([G, BS], accum_dtype)
            # acc_s_cast: fp16 probs for fast GEMM with V
            acc_s_cast = T.alloc_fragment([G, BS], dtype)
            # acc_o: [G, BV] - output accumulator tile (numerator of attention)
            acc_o = T.alloc_fragment([G, BV], accum_dtype)

            # Streaming softmax state per head (length G)
            scores_max = T.alloc_fragment([G], accum_dtype)       # running max
            scores_max_prev = T.alloc_fragment([G], accum_dtype)  # previous running max
            scores_scale = T.alloc_fragment([G], accum_dtype)     # rescale factor when max changes
            scores_sum = T.alloc_fragment([G], accum_dtype)       # sum of exp2(...) within a block
            logsum = T.alloc_fragment([G], accum_dtype)           # running denominator across blocks

            # Decode indices
            i_t, i_v, i_bh = bx, by, bz
            i_b, i_h = i_bh // head_kv, i_bh % head_kv

            # Load Q vectors for the G query heads served by KV head i_h
            # Slice heads: [i_h*G : (i_h+1)*G]
            T.copy(Q[i_b, i_t, i_h * G : (i_h + 1) * G, :], Q_shared)

            # Initialize accumulators
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            # Iterate over S selected blocks for this token
            for si in T.Pipelined(S, num_stages=num_stages):

                # Convert block-id to token start index
                i_s = BlockIndices[i_b, i_t, i_h, si] * BS

                # ------------------------------------------------------------
                # Guard: only process valid blocks.
                # Original had: (i_s <= i_t and i_s >= 0)
                # Added explicit OOB guard: (i_s + BS <= seq_len)
                # This prevents K/V loads from reading past end.
                # ------------------------------------------------------------
                if (i_s <= i_t) and (i_s >= 0) and (i_s + BS <= seq_len):

                    # Load K block [i_s : i_s+BS] into shared
                    T.copy(K[i_b, i_s : i_s + BS, i_h, :], K_shared)

                    # Causal mask inside the block:
                    # If key position (i_s + j) > i_t, mask it to -inf.
                    if is_causal:
                        for gi, j in T.Parallel(G, BS):
                            acc_s[gi, j] = T.if_then_else(
                                i_t >= (i_s + j),
                                0,
                                -T.infinity(acc_s.dtype),
                            )
                    else:
                        # Non-causal: no mask bias, start at 0
                        T.clear(acc_s)

                    # Scores: acc_s += Q_shared @ K_shared^T
                    T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                    # -------------------------
                    # Streaming softmax update
                    # -------------------------
                    # Save previous max
                    T.copy(scores_max, scores_max_prev)

                    # Compute new max over scores in this block
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=True)

                    # Rescale factor for previous accumulators when max changes:
                    # scale_prev = exp2(prev_max*scale - new_max*scale)
                    for gi in T.Parallel(G):
                        scores_scale[gi] = T.exp2(scores_max_prev[gi] * scale - scores_max[gi] * scale)

                    # Convert scores -> exp2(score*scale - max*scale)
                    for gi, j in T.Parallel(G, BS):
                        acc_s[gi, j] = T.exp2(acc_s[gi, j] * scale - scores_max[gi] * scale)

                    # Sum exp2(...) within this block
                    T.reduce_sum(acc_s, scores_sum, dim=1)

                    # Update running denominator:
                    # logsum = logsum * scores_scale + scores_sum
                    for gi in T.Parallel(G):
                        logsum[gi] = logsum[gi] * scores_scale[gi] + scores_sum[gi]

                    # Cast probs to fp16 for V GEMM
                    T.copy(acc_s, acc_s_cast)

                    # Rescale output accumulator by scores_scale (same reason as logsum rescale)
                    for gi, j in T.Parallel(G, BV):
                        acc_o[gi, j] *= scores_scale[gi]

                    # Load V block tile (only BV channels for this i_v tile)
                    T.copy(V[i_b, i_s : i_s + BS, i_h, i_v * BV : (i_v + 1) * BV], V_shared)

                    # Output accumulate: acc_o += probs @ V
                    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            # Final normalization: divide numerator by denominator
            eps = T.cast(1e-6, accum_dtype)
            for gi, j in T.Parallel(G, BV):
                acc_o[gi, j] /= (logsum[gi] + eps)

            # Store output tile
            T.copy(acc_o, O_shared)
            T.copy(
                O_shared,
                Output[i_b, i_t, i_h * G : (i_h + 1) * G, i_v * BV : (i_v + 1) * BV],
            )

    # Return the compiled prim_func
    return native_sparse_attention


def main():
    # Test config
    B, SEQ_LEN, H, HQ, D, S, block_size, dtype, scale = 2, 64, 1, 16, 32, 1, 32, torch.float16, 0.1

    # Compile kernel for this configuration
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

    

    # Create inputs
    torch.random.manual_seed(0)
    Q = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device="cuda")
    K = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device="cuda")
    V = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device="cuda")

    # Sparse indices: sentinel is SEQ_LEN (invalid); guard should skip it
    block_indices = torch.full((B, SEQ_LEN, H, S), SEQ_LEN, dtype=torch.long, device="cuda")
    for b in range(B):
        for t in range(SEQ_LEN):
            for h in range(H):
                i_i = torch.randperm(max(1, (t // block_size)), device="cuda")[:S]
                block_indices[b, t, h, : len(i_i)] = i_i

    block_indices = block_indices.sort(-1)[0].to(torch.int32)

    # Run kernel (async)
    # out = kernel(Q, K, V, block_indices)

    # 2) Force synchronization so we KNOW the kernel finished
    # torch.cuda.synchronize()

    out = kernel(Q, K, V, block_indices.to(torch.int32))
    torch.cuda.synchronize()
    print("ran kernel, out shape:", tuple(out.shape))
    print("sample:", out[0, 0, 0, :8].float().cpu())


if __name__ == "__main__":
    main()
