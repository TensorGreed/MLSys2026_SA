At a high level, for each query token t, instead of attending to all past keys 0..t, it attends only to a small list of selected key blocks (like DeepSeek/NSA style): selected_blocks = S.

Imports
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

The JIT wrapper
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
