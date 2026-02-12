At a high level, for each query token t, instead of attending to all past keys 0..t, it attends only to a small list of selected key blocks (like DeepSeek/NSA style): selected_blocks = S.

## Imports
```python
import torch
import tilelang
from tilelang import language as T
import tilelang.testing
from typing import Optional
```

- torch: creates tensors on GPU, and is the caller of the kernel.
- tilelang: JIT-compiles a CUDA kernel from Python code.
- tilelang.language as T: TileLang “DSL” primitives (T.copy, T.gemm, shared memory allocs, etc.).
- tilelang.testing.set_random_seed(0): makes random choices reproducible.

You also import einops but don’t use it in this snippet (can remove).

## The JIT wrapper
```python
@tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def native_sparse_attention(batch, heads, seq_len, dim, is_causal, scale=None, block_size=64, groups=1, selected_blocks=16):
```
This outer native_sparse_attention(...) is not the GPU kernel itself.
It’s a kernel generator: you pass compile-time constants like batch, seq_len, etc., and it returns a compiled kernel callable.

- out_idx=[-1]: “the output tensor is the last argument of the prim_func” (TileLang needs to know which argument(s) are outputs).
- pass_configs: compiler knobs
  - FAST_MATH: use faster approximate math (like exp approximations).
  - disables some advanced lowering/scheduling passes (TMA, warp-specialized) for simplicity or compatibility.

## Scaling factor (softmax math)
```python
if scale is None:
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
else:
    scale = scale * 1.44269504  # log2(e)
```

Attention uses softmax( (QK^T) * (1/sqrt(dim)) ).

This kernel uses exp2(x) (base-2 exponent) instead of exp(x) (base-e), so it multiplies by log2(e) ≈ 1.44269504 to convert:

- ```exp(x) == exp2(x * log2(e))```

So scale is pre-adjusted for exp2.

## Grouped-query attention shapes
```python
head_kv = heads // groups
q_shape = [batch, seq_len, heads, dim]
kv_shape = [batch, seq_len, head_kv, dim]
```
This is **GQA / grouped-query attention**:
- Q has ```heads```
- K/V have fewer heads: ```head_kv = heads / groups```

Each KV head is shared by ```groups``` query heads.

Example from your test:

- ```HQ = 16```, ```H = 1```, ```groups = HQ // H = 16```
- ```head_kv = heads // groups = 16 // 16 = 1```

So: 16 Q heads share 1 KV head.

## Sparse pattern input
```python
block_indices_shape = [batch, seq_len, head_kv, selected_blocks]
block_indices_dtype = T.int32
```

```BlockIndices[b, t, h, i]``` tells the kernel: for query token t, KV head h, attend to the i-th selected block (block id, not token id).

Important concept:

- ```BlockIndices``` stores **block indices** (0,1,2,...) — later multiplied by ```block_size``` to get token offset.

## Dtypes and tiling sizes
```python
dtype = T.float16
accum_dtype = T.float32
block_S = block_size
block_T = min(128, tilelang.math.next_power_of_2(dim))
```

- Q/K/V stored in FP16.
- Accumulators in FP32 for stability.

```block_S``` = number of tokens per sparse block (e.g., 64 or 32).

```block_T``` = how many channels (the dim) to process per tile.

- It rounds dim up to a power of 2, but caps at 128.

So if dim=32, block_T=32.

If dim=96, next_pow2=128, so block_T=128.

## Splitting dim into tiles
```python
NK = tilelang.cdiv(dim, block_T)
NV = tilelang.cdiv(dim, block_T)
assert NK == 1, "The key dimension can not be larger than 256"
````

- ```cdiv(a,b)``` = ceil(a/b)
- If ```dim <= block_T```, NK = 1
- This kernel currently only supports ```NK == 1``` (i.e. dim fits in one tile).

NV is used later as the grid dimension for V/output tiling.

Concept: the kernel produces output in **chunks of the hidden dimension**, so multiple CTAs cover different ```i_v``` tiles.

## Rename constants for readability
```python
S = selected_blocks
G = groups
BS = block_S
BK = BV = block_T
num_stages = 2
threads = 32
```
- S: number of selected blocks per query token.
- G: groups (how many query heads share one KV head).
- BS: block size in tokens.
- BK: tile size in channels for Q/K.
- BV: tile size in channels for V/O.

```threads=32```: one warp per CTA (simple design).
```num_stages=2```: pipeline depth for memory/compute overlap (double buffering).

## The real GPU kernel (TileLang prim_func)
```python
@T.prim_func
def native_sparse_attention(
    Q: T.Tensor(q_shape, dtype),
    K: T.Tensor(kv_shape, dtype),
    V: T.Tensor(kv_shape, dtype),
    BlockIndices: T.Tensor(block_indices_shape, block_indices_dtype),
    Output: T.Tensor(q_shape, dtype),
):
```
This defines the kernel signature. TileLang compiles this into CUDA.

## Launch geometry (grid mapping)
```python
with T.Kernel(seq_len, NV, batch * head_kv, threads=threads) as (bx, by, bz):
```
This is the most important mapping:

Grid dimensions:

- bx in [0..seq_len-1] → query token index t
- by in [0..NV-1] → which output-V tile of the hidden dim
- bz in [0..batch*head_kv-1] → batch and KV head combined

So one CTA computes:

> output for (batch b, kv-head h) at token t, for one V tile i_v.

## Shared memory buffers
```python
Q_shared = T.alloc_shared([G, BK], dtype)
K_shared = T.alloc_shared([BS, BK], dtype)
V_shared = T.alloc_shared([BS, BV], dtype)
O_shared = T.alloc_shared([G, BV], dtype)
```
Shared memory is “fast on-chip scratchpad” shared by threads in the CTA.

- Q_shared: holds the query vectors for the G query heads in this group, for this token t.
    - shape [G, BK]
- K_shared: holds one key block [BS tokens, BK channels]
- V_shared: holds one value block [BS tokens, BV channels for this tile]
- O_shared: temporary output staging

Concept: Load global → shared → compute (GEMM) to reduce global memory traffic.

## Register fragments (per-thread registers)
```python
acc_s = T.alloc_fragment([G, BS], accum_dtype)
acc_s_cast = T.alloc_fragment([G, BS], dtype)
acc_o = T.alloc_fragment([G, BV], accum_dtype)
scores_max = T.alloc_fragment([G], accum_dtype)
scores_max_prev = T.alloc_fragment([G], accum_dtype)
scores_scale = T.alloc_fragment([G], accum_dtype)
scores_sum = T.alloc_fragment([G], accum_dtype)
logsum = T.alloc_fragment([G], accum_dtype)
```
“fragment” here means: stored in registers (fastest) and often aligned to TensorCore MMA usage.

- acc_s: attention scores for this block: Q @ K^T, shape [G, BS] (G heads, BS keys)
- acc_o: output accumulator for this tile: shape [G, BV]
- softmax bookkeeping:
    - scores_max: running max for numerical stability
    - scores_max_prev: previous max
    - scores_scale: how much to rescale previous partial sums when max changes
    - scores_sum: sum of exp2(scores - max)
    - logsum: running denominator across blocks (streaming softmax)

Concept: This is online softmax (streaming softmax), so you don’t need to store all scores for all blocks.

## Decode indices
```python
i_t, i_v, i_bh = bx, by, bz
i_b, i_h = i_bh // head_kv, i_bh % head_kv
```
- i_t = token t
- i_v = which V tile (0..NV-1)
- i_bh is flattened; recover:
    - i_b = batch index
    - i_h = kv head index
 
## Load query for this token into shared
```python
NS = S
T.copy(Q[i_b, i_t, i_h * G : (i_h + 1) * G, :], Q_shared)
```
This copies Q vectors for the G query heads that correspond to KV head i_h.

If G=16 and i_h=0, this loads heads 0:16.

## Initialize accumulators
```python
T.fill(acc_o, 0)
T.fill(logsum, 0)
T.fill(scores_max, -T.infinity(accum_dtype))
```
- Start output at 0
- softmax denominator accumulator = 0
- max starts at -inf

## Loop over selected sparse blocks (pipelined)
```python
for i in T.Pipelined(NS, num_stages=num_stages):
    i_s = BlockIndices[i_b, i_t, i_h, i] * BS
    if i_s <= i_t and i_s >= 0:
```
For each selected block i:

- ```BlockIndices[...]``` gives block id
- multiply by BS → convert to token offset i_s
- guard:
    - ```i_s <= i_t```: only attend to past tokens (causal-ish)
    - ```i_s >= 0```: sanity

⚠️ Missing guard: you also need ```i_s + BS <= seq_len``` to avoid OOB loads.

## Load keys for this block
```python
T.copy(K[i_b, i_s : i_s + BS, i_h, :], K_shared)
```
Load a ```BS x BK``` tile of K into shared.

## Causal mask inside a block
```python
if is_causal:
    for i, j in T.Parallel(G, BS):
        acc_s[i, j] = T.if_then_else(i_t >= (i_s + j), 0, -T.infinity(acc_s.dtype))
else:
    T.clear(acc_s)
```
This pre-fills acc_s with:

- 0 for allowed positions
- -inf for forbidden positions

Why?

Because later they do:
```python
T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, ...)
```
That GEMM adds into acc_s (conceptually ```acc_s += QK^T```), so starting at 0 means “no bias”, and starting at -inf means “masked out forever”.

If not causal, T.clear(acc_s) sets it to 0.

## Compute scores = QK^T
```python
T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
```
- Q_shared: [G, BK]
- K_shared: [BS, BK] but transposed → [BK, BS]
- result: [G, BS] stored/accumulated into acc_s

This is the attention score for this block.

## Streaming softmax
### Save previous max, compute new max
```python
T.copy(scores_max, scores_max_prev)
T.fill(scores_max, -T.infinity(accum_dtype))
T.reduce_max(acc_s, scores_max, dim=1, clear=True)
```
- scores_max_prev = previous max per head
- scores_max = max over the BS scores for each head in this block (plus previous blocks via streaming logic)

### Compute rescale factor
```python
for i in T.Parallel(G):
    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
```

This implements online softmax update:

If your running max changes from ```m_prev``` to ```m_new```,
then previous accumulated denominator and output must be rescaled by ```exp(m_prev - m_new)```.

Here in base-2 with scale.

### Exponentiate scores (shifted by max)
```python
for i, j in T.Parallel(G, BS):
    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
```

Now acc_s becomes ```exp2((score - max) * scale)``` which is stable.

### Sum exponentials
```python
T.reduce_sum(acc_s, scores_sum, dim=1)
```

```scores_sum[i]``` = sum over j of exp2-shifted scores for head i for this block.

### Update running denominator
```python
for i in T.Parallel(G):
    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
```

Streaming update:
```python
logsum_new = logsum_prev * rescale + block_sum
```
### Cast probs to fp16 for the V GEMM
```python
T.copy(acc_s, acc_s_cast)
```

This makes probabilities fp16 to use TensorCores efficiently in the next GEMM.
