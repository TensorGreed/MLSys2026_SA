At a high level, for each query token t, instead of attending to all past keys 0..t, it attends only to a small list of selected key blocks (like DeepSeek/NSA style): selected_blocks = S.

Imports
```python
import torch
import tilelang
from tilelang import language as T
import tilelang.testing
from typing import Optional
```
