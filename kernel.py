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
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MiniDSARouter(nn.Module):
    """
    Mini-DSA Router (Learned Projection + Learned Routing)
    ------------------------------------------------------

    Goal:
      Produce BlockIndices[b, t, kv_head, s] = which *key blocks* each query token should attend to.

    Key idea:
      Instead of comparing Q to *all keys* (T tokens),
      we compare Q to *block summaries* (num_blocks blocks),
      BUT we do the comparison in a *learned low-dimensional routing space*.

    This approximates "DeepSeek-like" dynamic sparse attention at a conceptual level:
      - We learn a routing space (small dim Dr)
      - We score (Q_routing dot K_routing) for each block
      - We pick Top-K blocks dynamically per token

    Important:
      This module does NOT change your TileLang kernel.
      It only produces the block_indices that you feed into the kernel.

    Shapes (common in your setup):
      Q: [B, T, HQ, D]   (many query heads)
      K: [B, T, H,  D]   (fewer KV heads in GQA)
      groups = HQ // H   (how many Q heads share one KV head)

    Router output:
      block_indices: [B, T, H, S]  (S selected blocks per token per KV head)
    """

    def __init__(self, dim: int, dr: int, kv_heads: int):
        """
        dim:     D, original head dimension (e.g., 64)
        dr:      Dr, routing dimension (small, e.g., 8, 16, 32)
                 Smaller = cheaper + more bottleneck (often good for routing)
                 Larger = potentially more accurate but more compute
        kv_heads: H, number of KV heads
        """
        super().__init__()

        # These are the learned projection matrices.
        #
        # For each KV head h:
        #   - Wq[h] maps a query vector from D -> Dr
        #   - Wk[h] maps a block-summary key vector from D -> Dr
        #
        # Using per-head matrices is more flexible (and still tiny).
        #
        # Shapes:
        #   Wq: [H, D, Dr]
        #   Wk: [H, D, Dr]
        self.Wq = nn.Parameter(torch.randn(kv_heads, dim, dr) * 0.02)
        self.Wk = nn.Parameter(torch.randn(kv_heads, dim, dr) * 0.02)

        # Optional learned temperature/scale per head.
        # Multiplying scores by exp(logit_scale[h]) lets each head learn
        # how sharp or soft its routing distribution should be.
        self.logit_scale = nn.Parameter(torch.zeros(kv_heads))

    @torch.no_grad()
    def hard_topk_blocks(
        self,
        Q: torch.Tensor,          # [B, T, HQ, D]
        K: torch.Tensor,          # [B, T, H,  D]
        block_size: int,
        selected_blocks: int,
        groups: int,
        is_causal: bool = True,
        add_local: int = 1,
        summary: str = "mean",    # keep mean/first/last options
    ) -> torch.Tensor:
        """
        Compute Top-K block indices per token (HARD routing).

        HARD = we use torch.topk(...) to choose discrete blocks.
        This is the "real" behavior you want at inference.

        Note:
          topk is not differentiable; that's why training typically uses a
          soft objective (e.g., KL to dense block-attention) and then you
          switch to hard topk at inference time.

        add_local:
          Adds local (current + previous) blocks to stabilize routing,
          because real sparse attention systems often keep a local window.
          Example add_local=1 means: union(topk_blocks, {t_block, t_block-1}).
        """
        device = Q.device
        B, T, HQ, D = Q.shape
        _, Tk, H, Dk = K.shape

        # Basic sanity checks
        assert Tk == T, "Q and K must have same sequence length"
        assert Dk == D, "Q and K must have same head dimension"
        assert HQ % H == 0, "HQ must be divisible by H for GQA"
        assert groups == (HQ // H), "groups must equal HQ//H"

        BS = block_size
        num_blocks = (T + BS - 1) // BS  # e.g., T=1024, BS=64 => 16 blocks

        # ------------------------------------------------------------
        # (1) Build ONE "summary key vector" per block, per kv head.
        # ------------------------------------------------------------
        #
        # We want to go from token-level keys:
        #   K:   [B, T, H, D]
        # to block-level summaries:
        #   Ksum:[B, num_blocks, H, D]
        #
        # This is the "compress each block into 1 vector" step.
        #
        # Why do this?
        #   Because routing should be cheap: compare Q to 16 blocks,
        #   not to 1024 tokens.
        pad = num_blocks * BS - T
        if pad > 0:
            # pad token dimension so we can reshape into exact blocks
            # F.pad uses (last_dim_left, last_dim_right, ..., first_dim_left, first_dim_right)
            # Here K is [B, T, H, D]; we pad on T dimension: (0,0) for D, (0,0) for H, (0,pad) for T
            Kp = F.pad(K, (0, 0, 0, 0, 0, pad))
        else:
            Kp = K

        # Reshape tokens into blocks:
        #   [B, Tpad, H, D] -> [B, num_blocks, BS, H, D]
        # Now BS is "tokens inside each block"
        Kb = Kp.view(B, num_blocks, BS, H, D)

        # Choose how to summarize each block into ONE vector:
        if summary == "mean":
            # Average all tokens in the block -> stable, smooth summary
            Ksum = Kb.mean(dim=2)         # [B, num_blocks, H, D]
        elif summary == "last":
            # Use last token in each block as representative
            Ksum = Kb[:, :, BS - 1, :, :] # [B, num_blocks, H, D]
        elif summary == "first":
            # Use first token in each block
            Ksum = Kb[:, :, 0, :, :]      # [B, num_blocks, H, D]
        else:
            raise ValueError(f"Unknown summary='{summary}'")

        # ------------------------------------------------------------
        # (2) Pick a representative Q head per KV head (GQA mapping)
        # ------------------------------------------------------------
        #
        # In GQA:
        #   - Many query heads (HQ)
        #   - Fewer KV heads (H)
        #   - groups = HQ // H
        #
        # KV head h corresponds to query head (h*groups) as a representative.
        # (This is a cheap approximation; you could also pool multiple Q heads.)
        rep_qh = torch.arange(H, device=device) * groups  # [H]
        Qrep = Q[:, :, rep_qh, :]                         # [B, T, H, D]

        # ------------------------------------------------------------
        # (3) Project Qrep and Ksum into the learned routing space Dr
        # ------------------------------------------------------------
        #
        # This is the key "learned routing" idea:
        #   Qr = Qrep @ Wq   (D -> Dr)
        #   Kr = Ksum @ Wk   (D -> Dr)
        #
        # Now comparisons happen in Dr dimensions instead of D dimensions.
        # Dr is usually much smaller, so routing is cheap.
        #
        # Shapes:
        #   Qr: [B, T, H, Dr]
        #   Kr: [B, NB, H, Dr]
        Qr = torch.einsum("bthd,hdr->bthr", Qrep.float(), self.Wq.float())
        Kr = torch.einsum("bnhd,hdr->bnhr", Ksum.float(), self.Wk.float())

        # ------------------------------------------------------------
        # (4) Score each token against each block summary (per head)
        # ------------------------------------------------------------
        #
        # scores[b, t, h, n] = dot( Qr[b,t,h,:], Kr[b,n,h,:] )
        #
        # Output shape:
        #   scores: [B, T, H, num_blocks]
        scores = torch.einsum("bthr,bnhr->bthn", Qr, Kr)

        # Optional head-wise temperature scaling (learned)
        scale = torch.exp(self.logit_scale).view(1, 1, H, 1).float()
        scores = scores * scale

        # ------------------------------------------------------------
        # (5) Apply causal legality at the BLOCK level
        # ------------------------------------------------------------
        #
        # If is_causal=True, token t can only attend to keys k <= t.
        # At block level, token t can only attend to blocks <= t_block
        # where t_block = floor(t / BS).
        #
        # This prevents selecting "future blocks".
        if is_causal:
            t_blk = (torch.arange(T, device=device) // BS).view(1, T, 1, 1)  # [1,T,1,1]
            blk_ids = torch.arange(num_blocks, device=device).view(1, 1, 1, num_blocks)
            scores = scores.masked_fill(blk_ids > t_blk, float("-inf"))

        # ------------------------------------------------------------
        # (6) Pick Top-K blocks (HARD routing)
        # ------------------------------------------------------------
        #
        # This is the discrete sparse pattern selection.
        # If selected_blocks is small (like 4 or 8), we get real sparsity.
        Kpick = min(selected_blocks, num_blocks)
        top = torch.topk(scores, k=Kpick, dim=-1).indices  # [B, T, H, Kpick]

        # ------------------------------------------------------------
        # (7) Add local blocks (stability / accuracy trick)
        # ------------------------------------------------------------
        #
        # In real systems, purely global Top-K can be unstable.
        # It's common to always include local blocks near the current token.
        #
        # add_local=1 means include:
        #   t_block and (t_block-1)
        #
        # This gives:
        #   top = union(topk_blocks, local_blocks)
        if add_local > 0 and is_causal:
            t_blk = (torch.arange(T, device=device) // BS).view(1, T, 1, 1)
            locals_ = []
            for d in range(add_local + 1):
                locals_.append(torch.clamp(t_blk - d, min=0))
            local = torch.cat(locals_, dim=-1)     # [1, T, 1, add_local+1]
            local = local.expand(B, T, H, -1)      # [B, T, H, add_local+1]
            top = torch.cat([top, local], dim=-1)  # [B, T, H, Kpick + add_local+1]

        # ------------------------------------------------------------
        # (8) Sort / (cheap) unique / trim to exactly S blocks
        # ------------------------------------------------------------
        #
        # We want stable deterministic ordering + remove duplicates introduced by local union.
        top_sorted, _ = torch.sort(top, dim=-1)

        # Cheap "unique" for small S: remove adjacent duplicates after sorting.
        uniq = []
        for i in range(top_sorted.shape[-1]):
            if i == 0:
                uniq.append(top_sorted[..., i:i+1])
            else:
                prev = uniq[-1][..., -1:]
                cur = top_sorted[..., i:i+1]
                keep = (cur != prev)
                uniq.append(torch.where(keep, cur, prev))

        uniq = torch.cat(uniq, dim=-1)
        uniq, _ = torch.sort(uniq, dim=-1)

        # Trim to exactly selected_blocks and return int32 for kernel
        out = uniq[..., :selected_blocks].contiguous().to(torch.int32)  # [B, T, H, S]
        return out


def dense_block_teacher(Qrep, K, BS, is_causal=True):
    """
    Build a "teacher" distribution over blocks using DENSE attention.

    Inputs:
      Qrep: [B, T, H, D]
        - Representative queries per KV head (GQA mapping already applied).
        - For KV head h, we use Q head (h*groups) as the query used for routing.

      K:    [B, T, H, D]
        - Full keys per KV head.

      BS: block_size (e.g., 64)
      is_causal: if True, tokens cannot attend to future keys (k > t)

    Output:
      P_blk: [B, T, H, NB]
        - For each (b,t,h), a probability distribution across blocks (NB blocks).
        - This is built by:
            dense attention probs across tokens -> sum probs within each block.
    """
    B, T, H, D = Qrep.shape
    assert K.shape == (B, T, H, D), f"K must be [B,T,H,D], got {tuple(K.shape)}"
    NB = (T + BS - 1) // BS

    # ------------------------------------------------------------
    # (1) Compute dense attention logits over TOKENS:
    #     logits[b,t,h,k] = dot(Qrep[b,t,h,:], K[b,k,h,:])
    #
    # Shape:
    #   Qrep: [B,T,H,D]
    #   K:    [B,T,H,D] (but we use token index as 'k')
    #   logits -> [B,T,H,T]
    # ------------------------------------------------------------
    logits = torch.einsum("bthd,bkhd->bthk", Qrep.float(), K.float())

    # ------------------------------------------------------------
    # (2) Apply causal mask: disallow k > t by setting logits=-inf
    # ------------------------------------------------------------
    if is_causal:
        t_idx = torch.arange(T, device=Qrep.device).view(1, T, 1, 1)  # [1,T,1,1]
        k_idx = torch.arange(T, device=Qrep.device).view(1, 1, 1, T)  # [1,1,1,T]
        logits = logits.masked_fill(k_idx > t_idx, float("-inf"))

    # ------------------------------------------------------------
    # (3) Softmax over token axis k -> dense attention probabilities
    #     P_tok[b,t,h,k] sums to 1 across k
    # ------------------------------------------------------------
    P_tok = torch.softmax(logits, dim=-1)  # [B,T,H,T]

    # ------------------------------------------------------------
    # (4) Convert token-level probs into block-level probs:
    #     - reshape token axis T into NB blocks of size BS
    #     - sum probs inside each block
    #
    # If T not divisible by BS, pad on token axis.
    # ------------------------------------------------------------
    pad = NB * BS - T
    if pad > 0:
        P_tok = F.pad(P_tok, (0, pad))  # pad last dim (token k)

    # After pad: token axis length = NB*BS
    # Reshape: [B,T,H,NB,BS], then sum over BS -> [B,T,H,NB]
    P_blk = P_tok.view(B, T, H, NB, BS).sum(dim=-1)

    # Normalize to be safe (numerical stability)
    P_blk = P_blk / (P_blk.sum(dim=-1, keepdim=True) + 1e-9)
    return P_blk


def router_logits_over_blocks(router, Q, K, BS, groups, summary="mean", is_causal=True):
    """
    Compute router *logits over blocks* (NO topk here).

    This is the differentiable part used for training:
      - we produce a distribution over blocks via softmax(logits)
      - we match it to the dense teacher distribution

    Inputs:
      Q: [B,T,HQ,D]
      K: [B,T,H,D]
      BS: block_size
      groups: HQ//H (e.g., 4)
      summary: how to summarize K inside each block ("mean"/"first"/"last")

    Outputs:
      logits_blk: [B,T,H,NB]
      Qrep:       [B,T,H,D]  (representative queries per KV head, used for teacher)
    """
    device = Q.device
    B, T, HQ, Dq = Q.shape
    Bk, Tk, H, Dk = K.shape
    assert Bk == B and Tk == T, "Q and K must share [B,T]"
    assert Dk == Dq, f"Q and K last dim must match; got Q={Dq}, K={Dk}"
    assert HQ % H == 0, "HQ must be divisible by H"
    assert groups == (HQ // H), "groups must equal HQ//H"

    NB = (T + BS - 1) // BS

    # ------------------------------------------------------------
    # (1) Summarize keys per block: Ksum [B,NB,H,D]
    # ------------------------------------------------------------
    pad = NB * BS - T
    if pad > 0:
        Kp = F.pad(K, (0, 0, 0, 0, 0, pad))  # pad token dimension
    else:
        Kp = K

    # Kb: [B,NB,BS,H,D]
    Kb = Kp.view(B, NB, BS, H, Dk)

    if summary == "mean":
        Ksum = Kb.mean(dim=2)          # [B,NB,H,D]
    elif summary == "first":
        Ksum = Kb[:, :, 0, :, :]       # [B,NB,H,D]
    elif summary == "last":
        Ksum = Kb[:, :, BS - 1, :, :]  # [B,NB,H,D]
    else:
        raise ValueError(summary)

    # ------------------------------------------------------------
    # (2) GQA mapping: select one representative Q head per KV head
    #
    # For KV head h, representative Q head index = h*groups
    # rep_qh: [H]
    # Qrep:  [B,T,H,D]
    # ------------------------------------------------------------
    rep_qh = torch.arange(H, device=device) * groups
    Qrep = Q[:, :, rep_qh, :]  # [B,T,H,D]

    # ------------------------------------------------------------
    # (3) Learned projection into routing space Dr
    #
    # router.Wq: [H,D,Dr]
    # router.Wk: [H,D,Dr]
    #
    # Qr: [B,T,H,Dr]
    # Kr: [B,NB,H,Dr]
    # ------------------------------------------------------------
    # IMPORTANT: If your Q/K last dimension is not what you expected,
    # this is where einsum will throw. Our asserts above catch it early.
    Qr = torch.einsum("bthd,hdr->bthr", Qrep.float(), router.Wq.float())
    Kr = torch.einsum("bnhd,hdr->bnhr", Ksum.float(), router.Wk.float())

    # ------------------------------------------------------------
    # (4) Similarity logits over blocks:
    # logits[b,t,h,n] = dot(Qr[b,t,h,:], Kr[b,n,h,:])
    # Shape: [B,T,H,NB]
    # ------------------------------------------------------------
    logits = torch.einsum("bthr,bnhr->bthn", Qr, Kr)

    # Optional learned head-wise scaling (temperature)
    logits = logits * torch.exp(router.logit_scale).view(1, 1, H, 1).float()

    # ------------------------------------------------------------
    # (5) Causal mask at BLOCK level
    # Disallow blocks > t_block (future blocks)
    # ------------------------------------------------------------
    if is_causal:
        t_blk = (torch.arange(T, device=device) // BS).view(1, T, 1, 1)
        blk = torch.arange(NB, device=device).view(1, 1, 1, NB)
        logits = logits.masked_fill(blk > t_blk, float("-inf"))

    return logits, Qrep


def train_router(router, Q, K, *, BS=64, groups=4, steps=400, lr=3e-3, summary="mean"):
    """
    Train router parameters (Wq, Wk, logit_scale) to match dense teacher block distribution.

    Training objective:
      KL( P_teacher_blocks || P_router_blocks )

    Why KL?
      - We want the router to assign high probability to blocks that dense attention
        actually uses (sum of attention mass in that block).
      - This yields a "learned routing policy" that is accuracy-oriented.

    Notes:
      - This trains only the router. Your TileLang kernel remains unchanged.
      - This is 'accurate': after training, sparse output should approximate dense output,
        so you might NOT see dramatic output differences—what changes is compute.
    """
    router.train()
    opt = torch.optim.AdamW(router.parameters(), lr=lr, weight_decay=1e-4)

    for step in range(steps):
        opt.zero_grad(set_to_none=True)

        # Router logits across blocks (differentiable)
        logits_blk, Qrep = router_logits_over_blocks(
            router, Q, K, BS, groups, summary=summary, is_causal=True
        )
        P_router = torch.softmax(logits_blk, dim=-1)  # [B,T,H,NB]

        # Dense teacher distribution (no grad)
        with torch.no_grad():
            P_teacher = dense_block_teacher(Qrep, K, BS, is_causal=True)  # [B,T,H,NB]

        # KL(P_teacher || P_router) over blocks, averaged
        # KL = sum_i P_teacher[i] * (log P_teacher[i] - log P_router[i])
        loss = torch.sum(
            P_teacher * (torch.log(P_teacher + 1e-9) - torch.log(P_router + 1e-9)),
            dim=-1
        ).mean()

        loss.backward()
        opt.step()

        # Basic logging: loss + average entropy of router distribution
        if step % 50 == 0 or step == steps - 1:
            with torch.no_grad():
                entropy = (-P_router * torch.log(P_router + 1e-9)).sum(dim=-1).mean().item()
            print(f"step {step:4d} | loss {loss.item():.6f} | router entropy {entropy:.3f}")

    router.eval()

# -----------------------------------------------------------------------------
# Helper: scaled dot product attention, written explicitly for clarity
# -----------------------------------------------------------------------------
def dense_attention_output(Qrep, K, V, is_causal=True, scale=None):
    """
    Compute FULL dense attention output for Qrep against K/V.

    Shapes:
      Qrep: [B, T, H, D]   (representative queries per KV head)
      K:    [B, T, H, D]
      V:    [B, T, H, D]

    Output:
      O:    [B, T, H, D]

    Notes:
      - This is the "teacher" output. It's expensive: O(T^2).
      - We keep it in float for numerical stability, then cast back if desired.
    """
    B, T, H, D = Qrep.shape
    assert K.shape == (B, T, H, D)
    assert V.shape == (B, T, H, D)

    # Default scale is 1/sqrt(D) like standard attention.
    if scale is None:
        scale = 1.0 / (D ** 0.5)

    # logits[b,t,h,k] = dot(Qrep[b,t,h,:], K[b,k,h,:])
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


# -----------------------------------------------------------------------------
# Differentiable "router-gated" attention:
# We do dense attention, BUT we add a block prior from the router.
# That makes the output depend on router parameters (Wq/Wk/logit_scale).
# -----------------------------------------------------------------------------
def block_gated_attention_output(Qrep, K, V, P_blocks, block_size, is_causal=True, scale=None, eps=1e-9):
    """
    Compute attention output where routing selects blocks SOFTLY (differentiable).

    Shapes:
      Qrep:    [B, T, H, D]
      K, V:    [B, T, H, D]
      P_blocks:[B, T, H, NB]   router probability over blocks for each token

    block_size (BS):
      - Tokens 0..BS-1 are in block 0
      - Tokens BS..2BS-1 are in block 1
      - etc.

    Output:
      O_gated: [B, T, H, D]

    The key idea:
      - compute token logits as usual: q·k
      - compute a per-token per-key "block bias" = log(P_blocks[t, block(k)])
      - add that bias to logits BEFORE softmax
      - now router influences which keys get probability mass (differentiably)

    Why log-space?
      - Multiplying probs is adding logs:
          softmax(qk) * P_block_prior  ->  logits + log(prior)
      - This behaves like a Bayesian prior over blocks.
    """
    B, T, H, D = Qrep.shape
    assert K.shape == (B, T, H, D)
    assert V.shape == (B, T, H, D)

    NB = P_blocks.shape[-1]
    assert P_blocks.shape == (B, T, H, NB)

    if scale is None:
        scale = 1.0 / (D ** 0.5)

    # ------------------------------------------------------------
    # (1) Dense token logits: [B,T,H,T]
    # ------------------------------------------------------------
    logits = torch.einsum("bthd,bkhd->bthk", Qrep.float(), K.float()) * scale

    # ------------------------------------------------------------
    # (2) Build "block id for each key token k"
    #     key_block[k] = k // BS
    #     shape: [T]
    # ------------------------------------------------------------
    key_block = (torch.arange(T, device=Qrep.device) // block_size).clamp(max=NB-1)  # [T]

    # ------------------------------------------------------------
    # (3) Convert block probabilities to log-priors:
    #     log_prior[b,t,h,k] = log(P_blocks[b,t,h, key_block[k]])
    #
    #     We gather along the block dimension using key_block.
    # ------------------------------------------------------------
    # P_blocks: [B,T,H,NB]
    # gather index must be broadcastable to [B,T,H,T]
    gather_index = key_block.view(1, 1, 1, T).expand(B, T, H, T)
    log_prior = torch.gather(P_blocks.clamp_min(eps).log(), dim=-1, index=gather_index)

    # ------------------------------------------------------------
    # (4) Add router prior into logits:
    #     If router thinks a block is important, log_prior is higher,
    #     increasing attention mass to tokens in that block.
    # ------------------------------------------------------------
    logits = logits + log_prior

    # ------------------------------------------------------------
    # (5) Causal mask remains valid
    # ------------------------------------------------------------
    if is_causal:
        t_idx = torch.arange(T, device=Qrep.device).view(1, T, 1, 1)
        k_idx = torch.arange(T, device=Qrep.device).view(1, 1, 1, T)
        logits = logits.masked_fill(k_idx > t_idx, float("-inf"))

    # ------------------------------------------------------------
    # (6) Softmax and output
    # ------------------------------------------------------------
    P = torch.softmax(logits, dim=-1)  # [B,T,H,T]
    O = torch.einsum("bthk,bkhd->bthd", P, V.float())
    return O


# -----------------------------------------------------------------------------
# Entropy regularization on router distributions
# -----------------------------------------------------------------------------
def router_entropy(P_blocks, eps=1e-9):
    """
    Compute mean entropy of router distribution over blocks.

    P_blocks: [B, T, H, NB]
    Entropy per (b,t,h):  -sum_n p[n] log p[n]

    Interpreting entropy:
      - High entropy  -> router is unsure / spreads mass across many blocks
      - Low entropy   -> router is confident / peaky (more sparse-like)

    For "sparsity", we usually ADD entropy as a penalty to minimize it.
    """
    P = P_blocks.clamp_min(eps)
    H = -(P * P.log()).sum(dim=-1)      # [B,T,H]
    return H.mean()


# -----------------------------------------------------------------------------
# JOINT training: match dense output (teacher) and encourage sparse routing
# -----------------------------------------------------------------------------
def train_router_joint(
    router,
    Q, K, V,
    *,
    BS=64,
    groups=None,
    steps=200,
    lr=3e-3,
    summary="mean",
    is_causal=True,
    alpha_block_kl=0.1,     # weight for block KL (optional but useful)
    beta_entropy=0.01,      # weight for entropy penalty (sparsity pressure)
    print_every=25
):
    """
    Train router with TWO key signals:

    (A) Output distillation (joint with attention output loss):
        Make router-gated attention output match dense attention output.

        - Teacher: dense_attention_output(Qrep,K,V)
        - Student: block_gated_attention_output(Qrep,K,V,P_router)

        Loss_out = MSE(student_output, teacher_output)

    (B) Block distribution matching (optional but very stabilizing):
        Compare router's block probs to teacher's block probs derived from dense attention.

        Loss_kl = KL(P_teacher_blocks || P_router_blocks)

    (C) Entropy regularization:
        Encourage router distribution to be peaky (lower entropy).
        Loss_ent = Entropy(P_router_blocks)

    Final:
        Loss = Loss_out + alpha*Loss_kl + beta*Loss_ent

    Notes:
      - This is "joint" in the sense that router is trained using OUTPUT mismatch,
        not only by matching distributions.
      - We are NOT backpropagating through TileLang kernel (hard indices).
        We use differentiable block-gated attention as the training surrogate.
    """
    device = Q.device
    B, T, HQ, Dq = Q.shape
    assert K.shape[:2] == (B, T)
    assert V.shape[:2] == (B, T)

    H = K.shape[2]
    Dk = K.shape[-1]
    assert Dq == Dk, f"Q/K dim mismatch: {Dq} vs {Dk}"
    assert HQ % H == 0, f"HQ({HQ}) must be divisible by H({H})"

    # groups = how many query heads share one KV head (GQA)
    if groups is None:
        groups = HQ // H
    assert groups == (HQ // H), "groups must equal HQ//H"

    # ------------------------------------------------------------
    # Representative Q per KV head:
    # For KV head h, we pick Q head index (h*groups).
    # This is exactly the mapping your sparse kernel assumes.
    # ------------------------------------------------------------
    rep_qh = torch.arange(H, device=device) * groups
    Qrep = Q[:, :, rep_qh, :]  # [B,T,H,D]

    router.train()
    opt = torch.optim.AdamW(router.parameters(), lr=lr, weight_decay=1e-4)

    # Fixed attention scale (standard)
    scale = 1.0 / (Dq ** 0.5)

    for step in range(steps):
        opt.zero_grad(set_to_none=True)

        # --------------------------------------------------------
        # 1) Router logits over blocks (differentiable)
        #    This should output logits_blk: [B,T,H,NB]
        # --------------------------------------------------------
        logits_blk, _ = router_logits_over_blocks(
            router, Q, K, BS, groups, summary=summary, is_causal=is_causal
        )

        # Convert logits -> probabilities over blocks
        P_router = torch.softmax(logits_blk, dim=-1)  # [B,T,H,NB]

        # --------------------------------------------------------
        # 2) Teacher: full dense attention output (expensive)
        # --------------------------------------------------------
        with torch.no_grad():
            O_teacher = dense_attention_output(Qrep, K, V, is_causal=is_causal, scale=scale)  # [B,T,H,D]

        # --------------------------------------------------------
        # 3) Student: router-gated attention output (differentiable)
        # --------------------------------------------------------
        O_student = block_gated_attention_output(
            Qrep, K, V,
            P_blocks=P_router,
            block_size=BS,
            is_causal=is_causal,
            scale=scale
        )  # [B,T,H,D]

        # --------------------------------------------------------
        # 4) Output distillation loss: match attention outputs
        # --------------------------------------------------------
        loss_out = F.mse_loss(O_student, O_teacher)

        # --------------------------------------------------------
        # 5) Optional: block teacher distribution + KL loss
        #    This stabilizes routing early in training.
        # --------------------------------------------------------
        if alpha_block_kl > 0:
            with torch.no_grad():
                P_teacher = dense_block_teacher(Qrep, K, BS, is_causal=is_causal)  # [B,T,H,NB]

            # KL(P_teacher || P_router)
            loss_kl = torch.sum(
                P_teacher * (torch.log(P_teacher + 1e-9) - torch.log(P_router + 1e-9)),
                dim=-1
            ).mean()
        else:
            loss_kl = torch.tensor(0.0, device=device)

        # --------------------------------------------------------
        # 6) Entropy regularization: encourage peaky routing
        # --------------------------------------------------------
        loss_ent = router_entropy(P_router)

        # --------------------------------------------------------
        # 7) Total loss
        # --------------------------------------------------------
        loss = loss_out + alpha_block_kl * loss_kl + beta_entropy * loss_ent
        loss.backward()
        opt.step()

        # --------------------------------------------------------
        # 8) Logging: show signals so you know training is real
        # --------------------------------------------------------
        if step % print_every == 0 or step == steps - 1:
            with torch.no_grad():
                ent = loss_ent.item()
                print(
                    f"step {step:4d} | "
                    f"loss={loss.item():.6f} | "
                    f"out_mse={loss_out.item():.6f} | "
                    f"blk_kl={loss_kl.item():.6f} | "
                    f"entropy={ent:.4f}"
                )

    router.eval()
    return router


# ==================================================================================================
# DEEPSEEK NSA: THREE-BRANCH ARCHITECTURE (arXiv:2502.11089)
# ==================================================================================================
#
# The actual DeepSeek "Native Sparse Attention" uses THREE parallel attention branches
# whose outputs are blended with learned per-head gating:
#
#   O = g_slc * O_selected + g_swa * O_window + (1 - g_slc - g_swa) * O_compressed
#
# Branch 1: COMPRESSED ATTENTION (global, coarse-grained)
#   - Compress K/V via strided 1D depthwise convolution with stride = compression_ratio
#   - Full attention on compressed token sequence (T/c tokens instead of T)
#   - Gives cheap global context (like reading chapter summaries of a book)
#
# Branch 2: SELECTED ATTENTION (sparse, fine-grained)
#   - Use importance scores (from compressed branch or a router) to pick top-k blocks
#   - Run the TileLang block-sparse kernel on selected blocks only
#   - This is the branch your existing kernel implements
#
# Branch 3: SLIDING WINDOW ATTENTION (local)
#   - Attend to a fixed window of recent tokens (e.g., last 512 tokens)
#   - Always included, prevents "shortcut learning" where the model
#     relies too heavily on compressed/selected and ignores local context
#   - Very important for coherence in text generation
#
# Gating:
#   - Two learned scalars per query-head per token: g_slc and g_swa
#   - These go through sigmoid (so they're in [0,1])
#   - Compressed branch gets the residual: g_cmp = 1 - g_slc - g_swa
#   - The model learns when to trust global vs. sparse vs. local context
#
# Why three branches?
#   - Compressed alone is too coarse (loses detail)
#   - Selected alone can miss recent context (top-k may skip nearby tokens)
#   - Window alone has no global reach
#   - Together they cover: global + important-blocks + local
#
# Reference: https://arxiv.org/abs/2502.11089
#            https://github.com/fla-org/native-sparse-attention
# ==================================================================================================


class CompressedAttention(nn.Module):
    """
    NSA Compressed Attention Branch (Branch 1)
    -------------------------------------------

    Purpose:
      Provide CHEAP GLOBAL CONTEXT by compressing the key/value sequences.

    How it works:
      1) Apply a 1D depthwise convolution with stride = compression_ratio (c) to K and V
         - This turns T tokens into approximately T/c "compressed tokens"
         - Each compressed token summarizes c consecutive original tokens
         - Depthwise conv means each head dimension is convolved independently
           (no cross-channel mixing, keeps it lightweight)

      2) Run standard causal attention of Q against the compressed K_c and V_c
         - Cost: O(T * T/c * D) instead of O(T * T * D) — c times cheaper

    Why depthwise convolution (not just mean pooling)?
      - Learned compression: the conv kernel learns WHAT to keep from each block
      - Mean pooling treats all positions equally; conv can weight them
      - The NSA paper uses this approach for better information retention

    Shapes:
      Input:  K, V:   [B, T, H, D]   (full resolution)
      Output: K_c, V_c: [B, T/c, H, D]   (compressed)
      Q stays at full resolution: [B, T, HQ, D]
      Attention output: [B, T, HQ, D]

    Parameters:
      compression_ratio (c): how many tokens to merge into one (default 4)
        - Higher c = more compression = cheaper but coarser
        - Lower c  = less compression = better quality but more expensive
      kernel_size: convolution window size (default = compression_ratio)
        - Typically equals c so each compressed token covers exactly c tokens
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
        Compress a tensor of shape [B, T, H, D] -> [B, T/c, H, D] using strided conv.

        Steps:
          1) Reshape [B, T, H, D] -> [B, T, H*D] -> [B, H*D, T] (Conv1d wants [B, C, T])
          2) Causal pad: add (kernel_size - 1) zeros on the LEFT so conv doesn't see future
          3) Apply strided conv -> [B, H*D, T/c]
          4) Reshape back -> [B, T/c, H, D]
        """
        B, T_len, H, D = X.shape

        # Step 1: Reshape for Conv1d
        # [B, T, H, D] -> [B, T, H*D] -> [B, H*D, T]
        X_flat = X.reshape(B, T_len, H * D).transpose(1, 2)  # [B, H*D, T]

        # Step 2: Causal padding
        # We pad on the LEFT with (kernel_size - 1) zeros
        # This ensures each output position only depends on past + current tokens
        pad_left = self.kernel_size - 1
        X_padded = F.pad(X_flat, (pad_left, 0))  # pad last dim (T) on the left

        return X_padded  # [B, H*D, T + pad_left]

    def forward(
        self,
        Q: torch.Tensor,          # [B, T, HQ, D]  (full resolution queries)
        K: torch.Tensor,          # [B, T, H,  D]  (full resolution keys)
        V: torch.Tensor,          # [B, T, H,  D]  (full resolution values)
        groups: int,              # HQ // H (GQA group count)
        is_causal: bool = True,
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Run compressed attention and return output [B, T, HQ, D].

        Also returns attention weights over compressed tokens (needed for indexer).
        """
        B, T_len, HQ, D = Q.shape
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
            # For query at position t, max allowed compressed index
            t_idx = torch.arange(T_len, device=Q.device).view(1, T_len, 1, 1)    # [1,T,1,1]
            tc_idx = torch.arange(T_c, device=Q.device).view(1, 1, 1, T_c)       # [1,1,1,T_c]
            # Compressed token tc covers original tokens up to (tc+1)*c - 1
            # Allow if (tc+1)*c - 1 <= t, i.e., tc < (t+1)/c
            causal_mask = ((tc_idx + 1) * c - 1) > t_idx  # True = FUTURE = MASK OUT
            logits = logits.masked_fill(causal_mask, float("-inf"))

        # -----------------------------------------------------------------------
        # (5) Softmax and weighted sum
        # -----------------------------------------------------------------------
        attn_weights = torch.softmax(logits, dim=-1)  # [B, T, HQ, T_c]
        O_compressed = torch.einsum("bths,bshd->bthd", attn_weights, V_c_exp.float())

        return O_compressed, attn_weights, K_c, V_c  # return attn_weights for indexer use


class SlidingWindowAttention(nn.Module):
    """
    NSA Sliding Window Attention Branch (Branch 3)
    -----------------------------------------------

    Purpose:
      Provide LOCAL CONTEXT by attending to a fixed window of recent tokens.

    How it works:
      For each query at position t, attend to tokens in [max(0, t - window_size + 1) .. t].
      This is standard causal attention, but limited to a fixed window.

    Why is this needed?
      - Selected attention (top-k blocks) might skip nearby tokens if they aren't
        in the "most important" blocks globally
      - But language is LOCAL: the most recent tokens are almost always relevant
      - This branch guarantees that local context is never lost
      - The NSA paper calls this "preventing shortcut learning"

    Performance:
      - Cost: O(T * window_size * D), linear in T for fixed window_size
      - Typically window_size = 512 or 1024

    Note:
      This is a PyTorch implementation (no custom kernel needed).
      For very long sequences, you'd want a Triton/CUDA kernel here too,
      but for this demo the PyTorch version is clear and correct.

    Shapes:
      Q: [B, T, HQ, D]
      K: [B, T, H,  D]
      V: [B, T, H,  D]
      Output: [B, T, HQ, D]
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
        K: torch.Tensor,          # [B, T, H,  D]
        V: torch.Tensor,          # [B, T, H,  D]
        groups: int,              # HQ // H
        is_causal: bool = True,
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Returns:
          O_window: [B, T, HQ, D]  attention output from sliding window
        """
        B, T_len, HQ, D = Q.shape
        H = K.shape[2]
        W = self.window_size

        if scale is None:
            scale = 1.0 / (D ** 0.5)

        # -----------------------------------------------------------------------
        # (1) Expand K, V for GQA (repeat KV heads to match Q heads)
        # -----------------------------------------------------------------------
        if groups > 1:
            K_exp = K.unsqueeze(3).expand(B, T_len, H, groups, D).reshape(B, T_len, HQ, D)
            V_exp = V.unsqueeze(3).expand(B, T_len, H, groups, D).reshape(B, T_len, HQ, D)
        else:
            K_exp = K
            V_exp = V

        # -----------------------------------------------------------------------
        # (2) Compute full attention logits, then mask to window
        # -----------------------------------------------------------------------
        # For simplicity and clarity, we compute full [T, T] logits and then mask.
        # In production, you'd use a windowed kernel to avoid O(T^2) memory.
        #
        # logits[b, t, hq, k] = Q[b,t,hq,:] . K[b,k,hq,:] * scale
        logits = torch.einsum("bthd,bkhd->bthk", Q.float(), K_exp.float()) * scale

        # -----------------------------------------------------------------------
        # (3) Create sliding window + causal mask
        # -----------------------------------------------------------------------
        # For each query position t, allow keys k where:
        #   (a) k <= t (causal: no future tokens)
        #   (b) k >= t - W + 1 (window: no tokens too far in the past)
        t_idx = torch.arange(T_len, device=Q.device).view(1, T_len, 1, 1)
        k_idx = torch.arange(T_len, device=Q.device).view(1, 1, 1, T_len)

        # Causal: mask out future
        causal_mask = k_idx > t_idx

        # Window: mask out tokens before the window
        window_mask = k_idx < (t_idx - W + 1)

        # Combined mask: either future or too far past
        combined_mask = causal_mask | window_mask
        logits = logits.masked_fill(combined_mask, float("-inf"))

        # -----------------------------------------------------------------------
        # (4) Softmax over allowed keys and compute weighted sum
        # -----------------------------------------------------------------------
        attn_probs = torch.softmax(logits, dim=-1)  # [B, T, HQ, T]
        O_window = torch.einsum("bthk,bkhd->bthd", attn_probs, V_exp.float())

        return O_window


@torch.no_grad()
def build_block_indices_from_compressed(
    attn_weights_compressed: torch.Tensor,  # [B, T, HQ, T_c]
    compression_ratio: int,
    block_size: int,
    selected_blocks: int,
    groups: int,
    add_local: int = 2,
) -> torch.Tensor:
    """
    NSA-Style Block Indexer: Derive block importance from compressed attention weights.

    This is how the real DeepSeek NSA chooses which blocks to attend to:
      1) The compressed attention branch computes attention weights over compressed tokens
      2) Each compressed token covers `compression_ratio` original tokens
      3) We map compressed attention mass back to original blocks
      4) Top-k blocks by accumulated attention mass = selected blocks

    Why this is better than a separately trained router:
      - No extra parameters to train
      - Importance is derived directly from actual attention patterns
      - The compressed branch already "knows" what's important
      - Simpler and more principled

    Inputs:
      attn_weights_compressed: [B, T, HQ, T_c]
        - Attention probabilities from compressed branch (already softmaxed)
        - For each query position t, this tells how much attention goes to
          each compressed token

      compression_ratio: c
        - Each compressed token covers c original tokens

      block_size: BS
        - Size of blocks in the selected attention kernel

      selected_blocks: S
        - How many blocks to select per token

      groups: HQ // H
        - GQA group count
        - We average across Q heads within each group to get per-KV-head scores

      add_local: int
        - Always include the most recent N blocks (same as in your existing indexer)

    Returns:
      block_indices: [B, T, H, S]  (int32)
        - Block IDs for the selected attention kernel
    """
    B, T_len, HQ, T_c = attn_weights_compressed.shape
    H = HQ // groups
    BS = block_size
    c = compression_ratio
    num_blocks = (T_len + BS - 1) // BS
    device = attn_weights_compressed.device

    # -----------------------------------------------------------------------
    # (1) Average attention weights across Q heads within each GQA group
    # -----------------------------------------------------------------------
    # attn_weights: [B, T, HQ, T_c] -> reshape to [B, T, H, groups, T_c] -> mean over groups
    # Result: [B, T, H, T_c]
    attn_per_kv = attn_weights_compressed.view(B, T_len, H, groups, T_c).mean(dim=3)

    # -----------------------------------------------------------------------
    # (2) Map compressed token attention to block importance
    # -----------------------------------------------------------------------
    # Each compressed token tc covers original tokens [tc*c .. (tc+1)*c - 1].
    # Each block b covers original tokens [b*BS .. (b+1)*BS - 1].
    # We need to sum attention mass from compressed tokens that overlap each block.
    #
    # Simple approach: for each compressed token tc, find which block(s) it overlaps.
    # Since c and BS may differ, one compressed token can span parts of multiple blocks.
    #
    # For simplicity, assign each compressed token to the block that contains its CENTER:
    #   center of tc = tc * c + c // 2
    #   block of center = center // BS
    block_scores = torch.zeros(B, T_len, H, num_blocks, device=device, dtype=torch.float32)

    # Map each compressed token index to a block index
    tc_indices = torch.arange(T_c, device=device)
    tc_centers = tc_indices * c + c // 2                # center position of each compressed token
    tc_to_block = (tc_centers // BS).clamp(max=num_blocks - 1)  # block ID for each compressed token

    # Scatter-add attention weights into block scores
    # attn_per_kv: [B, T, H, T_c]
    # tc_to_block: [T_c] -> expand to [1, 1, 1, T_c] -> [B, T, H, T_c]
    tc_block_expanded = tc_to_block.view(1, 1, 1, T_c).expand(B, T_len, H, T_c)
    block_scores.scatter_add_(dim=-1, index=tc_block_expanded, src=attn_per_kv)

    # -----------------------------------------------------------------------
    # (3) Apply causal mask at block level
    # -----------------------------------------------------------------------
    # Token t can only attend to blocks where block_start <= t
    t_blk = (torch.arange(T_len, device=device) // BS).view(1, T_len, 1, 1)
    blk_ids = torch.arange(num_blocks, device=device).view(1, 1, 1, num_blocks)
    block_scores = block_scores.masked_fill(blk_ids > t_blk, float("-inf"))

    # -----------------------------------------------------------------------
    # (4) Force-include local blocks (same technique as your existing indexer)
    # -----------------------------------------------------------------------
    if add_local and add_local > 0:
        for j in range(add_local):
            local_id = t_blk - j
            valid = (local_id >= 0)
            idx = local_id.clamp(min=0).expand(B, T_len, H, 1)
            boost = valid.expand(B, T_len, H, 1).float() * 1e9
            block_scores.scatter_add_(dim=-1, index=idx, src=boost)

    # -----------------------------------------------------------------------
    # (5) Top-k selection
    # -----------------------------------------------------------------------
    k = min(selected_blocks, num_blocks)
    top = torch.topk(block_scores, k=k, dim=-1).indices  # [B, T, H, k]
    top, _ = torch.sort(top, dim=-1)  # sort ascending for deterministic order

    return top.to(torch.int32)


class NativeSparseAttention(nn.Module):
    """
    DeepSeek NSA: Full Three-Branch Sparse Attention Module
    ========================================================

    This module implements the complete NSA architecture from arXiv:2502.11089.
    It orchestrates three parallel attention branches and blends their outputs
    using learned per-head gating.

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
                           where g_c = 1 - g_s - g_w

    Branch details:
      1) Compressed: Strided conv on K/V -> standard attention on shorter sequence
      2) Selected: Top-k block selection -> TileLang sparse kernel
      3) Sliding Window: Fixed-size local window attention

    Gating:
      - g_slc (gate for selected branch): learned from Q via linear projection + sigmoid
      - g_swa (gate for sliding window branch): same architecture
      - g_cmp = 1 - g_slc - g_swa (residual gate for compressed branch)
      - Each gate is [B, T, HQ] -> broadcast to [B, T, HQ, D]

    Usage:
      nsa = NativeSparseAttention(dim=64, kv_heads=4, q_heads=16, ...)
      O = nsa(Q, K, V, kernel_fn)  # kernel_fn is the compiled TileLang kernel

    Note:
      - The TileLang kernel is passed in externally (it's compiled once for a specific config)
      - This module handles everything else: compression, indexing, windowing, gating
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
        # For each of the two non-residual branches (selected, window),
        # we learn a gate from the query.
        #
        # Gate input: Q reshaped to [B, T, HQ * D]
        # Gate output: [B, T, HQ] -> one scalar per head per token
        #
        # We use a small MLP to avoid the gate being too simplistic.
        # Architecture: Linear(HQ*D -> HQ) with sigmoid activation
        #
        # Note: In the DeepSeek paper, gates are conditioned on the query token
        # representation. We use Q directly since we don't have a separate
        # pre-attention representation.
        #
        # Using per-head linear (more parameter-efficient) instead of full HQ*D -> HQ:
        # Each head gets its own scalar gate from its own D-dimensional query.
        self.gate_slc = nn.Linear(dim, 1, bias=True)   # per-head gate for selected branch
        self.gate_swa = nn.Linear(dim, 1, bias=True)   # per-head gate for window branch

        # Initialize gates so that initially all branches contribute roughly equally
        # sigmoid(0) = 0.5, so init bias to make g_slc ~ 0.33, g_swa ~ 0.33
        # sigmoid(-0.4) ≈ 0.4, which gives g_cmp ≈ 0.2 (compressed gets less initially,
        # which is reasonable since it's the coarsest branch)
        nn.init.zeros_(self.gate_slc.weight)
        nn.init.constant_(self.gate_slc.bias, -0.4)
        nn.init.zeros_(self.gate_swa.weight)
        nn.init.constant_(self.gate_swa.bias, -0.4)

    def forward(
        self,
        Q: torch.Tensor,              # [B, T, HQ, D]
        K: torch.Tensor,              # [B, T, H,  D]
        V: torch.Tensor,              # [B, T, H,  D]
        kernel_fn=None,               # compiled TileLang kernel (optional)
        is_causal: bool = True,
        scale: Optional[float] = None,
    ) -> dict:
        """
        Run the full three-branch NSA and return the gated output.

        Args:
          Q, K, V: query, key, value tensors
          kernel_fn: compiled TileLang kernel for selected attention (optional).
                     If None, selected branch uses PyTorch fallback.
          is_causal: whether to apply causal masking
          scale: attention scale (default: 1/sqrt(D))

        Returns:
          dict with keys:
            'output':        [B, T, HQ, D]  final gated NSA output
            'O_compressed':  [B, T, HQ, D]  compressed branch output
            'O_selected':    [B, T, HQ, D]  selected branch output
            'O_window':      [B, T, HQ, D]  sliding window branch output
            'g_slc':         [B, T, HQ]     selected gate values
            'g_swa':         [B, T, HQ]     window gate values
            'block_indices': [B, T, H, S]   selected block indices
        """
        B, T_len, HQ, D = Q.shape
        H = K.shape[2]
        groups = self.groups

        if scale is None:
            scale = 1.0 / (D ** 0.5)

        # ===================================================================
        # BRANCH 1: Compressed Attention
        # ===================================================================
        # This runs first because its attention weights are used to derive
        # block importance for the selected attention branch.
        O_compressed, attn_weights_c, K_c, V_c = self.compressed_attn(
            Q, K, V, groups=groups, is_causal=is_causal, scale=scale
        )

        # ===================================================================
        # INDEXER: Derive block indices from compressed attention
        # ===================================================================
        # The compressed attention weights tell us which parts of the sequence
        # are most relevant. We map these back to blocks for the sparse kernel.
        block_indices = build_block_indices_from_compressed(
            attn_weights_compressed=attn_weights_c,
            compression_ratio=self.compression_ratio,
            block_size=self.block_size,
            selected_blocks=self.selected_blocks,
            groups=groups,
            add_local=self.add_local,
        )

        # ===================================================================
        # BRANCH 2: Selected Attention (Sparse, via TileLang kernel)
        # ===================================================================
        if kernel_fn is not None:
            # Use the compiled TileLang kernel for maximum performance
            O_selected = kernel_fn(Q, K, V, block_indices)
        else:
            # PyTorch fallback: gather selected blocks and do attention
            # This is slower but doesn't require TileLang compilation
            O_selected = self._selected_attention_fallback(
                Q, K, V, block_indices, groups=groups, is_causal=is_causal, scale=scale
            )

        # Cast to float for gating math (kernel output is fp16)
        O_selected = O_selected.float()

        # ===================================================================
        # BRANCH 3: Sliding Window Attention
        # ===================================================================
        O_window = self.window_attn(
            Q, K, V, groups=groups, is_causal=is_causal, scale=scale
        )

        # ===================================================================
        # GATING: Blend branch outputs
        # ===================================================================
        # Compute per-head gates from Q
        # Q: [B, T, HQ, D]
        # gate_slc: Linear(D -> 1) applied per head
        # g_slc: [B, T, HQ]

        g_slc = torch.sigmoid(self.gate_slc(Q.float()).squeeze(-1))  # [B, T, HQ]
        g_swa = torch.sigmoid(self.gate_swa(Q.float()).squeeze(-1))  # [B, T, HQ]

        # Ensure g_slc + g_swa <= 1 so compressed branch gets non-negative weight
        # We clamp the sum and renormalize if needed
        gate_sum = g_slc + g_swa
        excess = (gate_sum > 1.0).float()
        # Where sum > 1, scale both gates down proportionally
        safe_sum = gate_sum.clamp(min=1e-6)
        g_slc = torch.where(gate_sum > 1.0, g_slc / safe_sum, g_slc)
        g_swa = torch.where(gate_sum > 1.0, g_swa / safe_sum, g_swa)

        # Compressed branch gets the residual
        g_cmp = 1.0 - g_slc - g_swa  # [B, T, HQ]

        # Expand gates for broadcasting with [B, T, HQ, D]
        g_slc_4d = g_slc.unsqueeze(-1)  # [B, T, HQ, 1]
        g_swa_4d = g_swa.unsqueeze(-1)  # [B, T, HQ, 1]
        g_cmp_4d = g_cmp.unsqueeze(-1)  # [B, T, HQ, 1]

        # Final gated combination:
        #   O = g_cmp * O_compressed + g_slc * O_selected + g_swa * O_window
        O_final = (
            g_cmp_4d * O_compressed.float()
            + g_slc_4d * O_selected.float()
            + g_swa_4d * O_window.float()
        )

        return {
            "output": O_final,
            "O_compressed": O_compressed,
            "O_selected": O_selected,
            "O_window": O_window,
            "g_slc": g_slc,
            "g_swa": g_swa,
            "g_cmp": g_cmp,
            "block_indices": block_indices,
        }

    def _selected_attention_fallback(
        self,
        Q: torch.Tensor,         # [B, T, HQ, D]
        K: torch.Tensor,         # [B, T, H,  D]
        V: torch.Tensor,         # [B, T, H,  D]
        block_indices: torch.Tensor,  # [B, T, H, S]
        groups: int,
        is_causal: bool = True,
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        PyTorch fallback for selected (block-sparse) attention.

        This does the same thing as the TileLang kernel but in pure PyTorch.
        Much slower, but useful when TileLang is not available (e.g., CPU testing).

        For each token t:
          1) Look up which blocks are selected (block_indices[b, t, h, :])
          2) Gather the K/V tokens from those blocks
          3) Run standard attention over the gathered tokens
        """
        B, T_len, HQ, D = Q.shape
        H = K.shape[2]
        S = block_indices.shape[-1]
        BS = self.block_size

        if scale is None:
            scale = 1.0 / (D ** 0.5)

        # This fallback operates per KV-head for simplicity
        # It's not optimized — just correct.
        O = torch.zeros_like(Q, dtype=torch.float32)

        for b_idx in range(B):
            for h_idx in range(H):
                # Q heads for this KV head: h_idx*groups .. (h_idx+1)*groups
                q_heads_range = slice(h_idx * groups, (h_idx + 1) * groups)
                Q_h = Q[b_idx, :, q_heads_range, :]  # [T, G, D]

                for t in range(T_len):
                    q_t = Q_h[t]  # [G, D]
                    blocks = block_indices[b_idx, t, h_idx].tolist()

                    # Gather K/V from selected blocks
                    k_tokens = []
                    v_tokens = []
                    positions = []
                    for blk_id in blocks:
                        start = blk_id * BS
                        end = min(start + BS, T_len)
                        if start < 0 or start >= T_len:
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
                        future_mask = pos > t
                        scores[:, future_mask] = float("-inf")

                    # Softmax and weighted sum
                    probs = torch.softmax(scores, dim=-1)
                    out_t = torch.matmul(probs, v_cat.float())  # [G, D]
                    O[b_idx, t, q_heads_range, :] = out_t

        return O


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

    # Fill selected blocks (clamp to valid range just in case)
    blk = idx.to("cpu").clamp(min=0, max=num_blocks - 1)  # [T_window, S]
    for t in range(blk.shape[0]):
        M[t, blk[t].tolist()] = 1.0

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

    # ==================================================================
    # (A) LEGACY PATH: MiniDSARouter + TileLang kernel
    # ==================================================================
    # This is your original code. We keep it as an alternative routing option.
    # It trains a small router (Wq/Wk projections) to match dense attention
    # block distributions, then uses hard top-k to pick blocks.
    print("\n" + "=" * 80)
    print("(A) LEGACY PATH: MiniDSARouter + TileLang Kernel")
    print("=" * 80)

    # Derive dims from actual tensors (safer than using local variables)
    HQ_actual = Q.shape[2]
    H_actual  = K.shape[2]
    D_actual  = Q.shape[-1]
    groups_actual = HQ_actual // H_actual

    router = MiniDSARouter(dim=D_actual, dr=16, kv_heads=H_actual).to("cuda")

    # Joint train (output loss + entropy)
    print("Training router (joint output + KL + entropy)...")
    train_router_joint(
        router, Q, K, V,
        BS=block_size,
        groups=groups_actual,
        steps=200,
        lr=3e-3,
        summary="mean",
        alpha_block_kl=0.1,
        beta_entropy=0.01,
    )

    # Get block indices from trained router
    block_indices_legacy = router.hard_topk_blocks(
        Q, K,
        block_size=block_size,
        selected_blocks=S,
        groups=groups_actual,
        is_causal=True,
        add_local=1,
    )

    # Run TileLang kernel
    out_legacy = kernel(Q, K, V, block_indices_legacy)

    print(f"\nLegacy output shape: {tuple(out_legacy.shape)}")
    print(f"Legacy finite:      {torch.isfinite(out_legacy).all().item()}")
    print(f"Legacy mean/std:    {out_legacy.float().mean().item():.4f} / {out_legacy.float().std().item():.4f}")

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

    # Run the full NSA pipeline
    # We pass the compiled TileLang kernel for the selected branch
    with torch.no_grad():
        result = nsa(Q, K, V, kernel_fn=kernel, is_causal=True, scale=scale)

    # Extract outputs
    O_nsa       = result["output"]
    O_compressed = result["O_compressed"]
    O_selected   = result["O_selected"]
    O_window     = result["O_window"]
    g_slc        = result["g_slc"]
    g_swa        = result["g_swa"]
    g_cmp        = result["g_cmp"]
    block_indices_nsa = result["block_indices"]

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
    print(f"  Sum check (should be ~1.0): {(g_slc + g_swa + g_cmp).mean().item():.6f}")

    # --- Block selection from NSA indexer ---
    print("\n--- NSA Block Selection (from compressed attention weights) ---")
    b, h = 0, 0
    print("Selected blocks (b=0, kv_head=0) for a few tokens:")
    for t in [0, 1, 15, 31, 32, 33, 63, 127, 255, min(511, SEQ_LEN - 1)]:
        if t < SEQ_LEN:
            print(f"  t={t:>3}  blk_ids={block_indices_nsa[b, t, h].cpu().tolist()}")

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
        _ = kernel(Q, K, V, block_indices_nsa)
    torch.cuda.synchronize()

    # Timed run
    iters = 200
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    start_evt.record()
    for _ in range(iters):
        _ = kernel(Q, K, V, block_indices_nsa)
    end_evt.record()

    torch.cuda.synchronize()
    ms = start_evt.elapsed_time(end_evt)
    print(f"  TileLang kernel: {ms/iters:.4f} ms/iter (avg over {iters} iters)")

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
