from __future__ import annotations

from dataclasses import dataclass, field

import torch


class CacheCapacityError(RuntimeError):
    pass


@dataclass
class Sequence:
    blocks: list[int] = field(default_factory=list)
    length: int = 0


class PagedKVCache:
    """Fixed-size KV pool with per-request logical-to-physical block tables."""

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_heads: int,
        head_dim: int,
        device: torch.device | str = "cpu",
    ) -> None:
        if min(num_blocks, block_size, num_heads, head_dim) <= 0:
            raise ValueError("cache dimensions must be positive")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = torch.device(device)
        shape = (num_blocks, block_size, num_heads, head_dim)
        self.keys = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.values = torch.zeros_like(self.keys)
        self.sequences: dict[int, Sequence] = {}
        self.free_blocks = list(range(num_blocks))
        self._clock = 0

    def create_sequence(self, request_id: int) -> None:
        if request_id in self.sequences:
            raise ValueError(f"request {request_id} already exists")
        self.sequences[request_id] = Sequence()
        self._touch(request_id)

    def _touch(self, request_id: int) -> None:
        self._clock += 1
        self.sequences[request_id].last_used = self._clock

    def append(self, request_id: int, keys: torch.Tensor, values: torch.Tensor) -> None:
        if request_id not in self.sequences:
            raise KeyError(f"unknown request {request_id}")
        expected = (keys.shape[0], self.num_heads, self.head_dim)
        if keys.ndim != 3 or values.shape != keys.shape or keys.shape[1:] != expected[1:]:
            raise ValueError(
                "expected keys and values with shape "
                f"[tokens, {self.num_heads}, {self.head_dim}]"
            )
        sequence = self.sequences[request_id]
        for index in range(keys.shape[0]):
            position = sequence.length
            slot = position % self.block_size
            if slot == 0:
                if not self.free_blocks:
                    raise CacheCapacityError("no free KV blocks")
                sequence.blocks.append(self.free_blocks.pop())
            physical = sequence.blocks[-1]
            self.keys[physical, slot].copy_(keys[index])
            self.values[physical, slot].copy_(values[index])
            sequence.length += 1
        self._touch(request_id)

    def block_table(self, request_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        if not request_ids:
            raise ValueError("at least one request is required")
        sequences = [self.sequences[request_id] for request_id in request_ids]
        width = max(1, max(len(sequence.blocks) for sequence in sequences))
        table = torch.full((len(request_ids), width), -1, dtype=torch.int64)
        lengths = torch.tensor([sequence.length for sequence in sequences], dtype=torch.int64)
        for row, sequence in enumerate(sequences):
            if sequence.blocks:
                table[row, : len(sequence.blocks)] = torch.tensor(sequence.blocks)
        return table.to(self.device), lengths.to(self.device)

    def contiguous_kv(self, request_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = self.sequences[request_id]
        keys = self.keys[sequence.blocks].reshape(-1, self.num_heads, self.head_dim)
        values = self.values[sequence.blocks].reshape(-1, self.num_heads, self.head_dim)
        return keys[: sequence.length].clone(), values[: sequence.length].clone()
