from __future__ import annotations

import os

import torch

if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")

import triton  # noqa: E402
import triton.language as tl  # noqa: E402

__all__ = ["tl", "triton"]
