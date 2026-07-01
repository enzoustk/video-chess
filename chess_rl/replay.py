"""Experience Replay — buffer circular.

Armazena apenas a **RAM crua** (128 bytes uint8) em vez do estado codificado,
o que reduz drasticamente o uso de memória (128 B vs ~3 KB por transição). A
codificação RAM -> (planos, auxiliar) é feita de forma vetorizada na hora da
amostragem (ver ``encoding.encode_batch``).
"""
from __future__ import annotations

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, ram_dim: int = 128, seed: int = 0):
        self.capacity = capacity
        self.ram = np.zeros((capacity, ram_dim), dtype=np.uint8)
        self.next_ram = np.zeros((capacity, ram_dim), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(self, ram, action, reward, next_ram, done):
        i = self.pos
        self.ram[i] = ram
        self.next_ram[i] = next_ram
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = self.rng.integers(0, self.size, size=batch_size)
        return (self.ram[idx], self.actions[idx], self.rewards[idx],
                self.next_ram[idx], self.dones[idx])

    def __len__(self):
        return self.size
