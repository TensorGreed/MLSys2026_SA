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

    # router = MiniDSARouter(dim=D, dr=16, kv_heads=H).to("cuda")
    SEQ_LEN=1024; BS=64; H=8; HQ=32; D=64; groups=HQ//H  # 4
    print("Q shape:", tuple(Q.shape))
    print("K shape:", tuple(K.shape))
    assert Q.shape[-1] == K.shape[-1]

    # Derive H/HQ/groups from tensors (avoid notebook drift)
    HQ_actual = Q.shape[2]
    H_actual  = K.shape[2]
    groups = HQ_actual // H_actual
    
    D_actual = Q.shape[-1]
    router = MiniDSARouter(dim=D_actual, dr=16, kv_heads=H_actual).to("cuda")
    
    # Joint train (output loss + entropy)
    train_router_joint(
        router, Q, K, V,
        BS=block_size,
        groups=groups,
        steps=200,           # start small
        lr=3e-3,
        summary="mean",
        alpha_block_kl=0.1,  # stabilizer
        beta_entropy=0.01,   # sparsity pressure
    )

    block_indices = router.hard_topk_blocks(
        Q, K, 
        block_size=BS, 
        selected_blocks=8, 
        groups=groups, 
        is_causal=True, 
        add_local=1)
    out = kernel(Q, K, V, block_indices)
    # after you create Q and K:
    # block_indices = router.hard_topk_blocks(
    #     Q=Q, K=K,
    #     block_size=block_size,
    #     selected_blocks=S,
    #     groups=HQ // H,
    #     is_causal=True,
    #     add_local=1,
    # )

    # This one is not learned
    # Build content-based Top-K block indices
    # block_indices = build_block_indices_topk(
    #     Q=Q,
    #     K=K,
    #     block_size=block_size,
    #     selected_blocks=S,
    #     groups=groups,
    #     summary="mean",
    #     add_local=2,  # always include last 2 blocks
    # )

    # Show selected blocks for a few tokens
    b, h = 0, 0
    print("\nSelected blocks (b=0, kv_head=0) for a few tokens:")
    for t in [0, 1, 15, 31, 32, 33, 63, 127, 255, 511]:
        print(f"t={t:>3}  blk_ids={block_indices[b, t, h].cpu().tolist()}")

    # Visualize selection for first 128 tokens
    ascii_blockmap(block_indices, BS=block_size, T_show=128, b=0, h=0)

    T = block_indices.shape[1]
    start_t = max(0, T - 256)
    
    plot_block_selection_heatmap(
        block_indices,
        block_size=block_size,
        b=0, h=0,
        start_t=start_t,
        T_show=min(256, T),
        title="Last tokens: selected blocks"
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
