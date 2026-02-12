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
