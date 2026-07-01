"""Arquitetura da rede neural profunda: Dueling DQN híbrida (CNN + MLP),
opcionalmente com **NoisyNets** (exploração paramétrica) nas cabeças e no trunk.

* O **tabuleiro** (12x8x8) passa por uma CNN — a convolução é natural para a
  estrutura espacial 2D do tabuleiro de xadrez.
* As **features auxiliares** (cursor/UI, 64 dims) passam por um MLP.
* As duas representações são concatenadas e a cabeça é **Dueling**:
  ``Q(s,a) = V(s) + (A(s,a) - média_a A(s,a))``, que estabiliza o aprendizado
  ao separar o valor do estado da vantagem de cada ação.
* Com ``noisy=True``, as Linears do trunk/value/advantage viram **NoisyLinear**:
  a exploração passa a ser via ruído paramétrico aprendido (descarta ε-greedy).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .encoding import AUX_LEN, BOARD_SHAPE
from .noisy import NoisyLinear


class DuelingDQN(nn.Module):
    def __init__(self, n_actions: int, n_planes: int = BOARD_SHAPE[0],
                 n_aux: int = AUX_LEN, hidden: int = 256, noisy: bool = False,
                 sigma_init: float = 0.5):
        super().__init__()
        self.noisy = noisy
        Lin = (lambda i, o: NoisyLinear(i, o, sigma_init=sigma_init)) if noisy else nn.Linear
        self.conv = nn.Sequential(
            nn.Conv2d(n_planes, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
        )
        conv_out = 64 * BOARD_SHAPE[1] * BOARD_SHAPE[2]  # 64*8*8 = 4096
        self.board_fc = nn.Sequential(nn.Linear(conv_out, hidden), nn.ReLU())
        self.aux_fc = nn.Sequential(nn.Linear(n_aux, 64), nn.ReLU())
        self.trunk_lin = Lin(hidden + 64, hidden)
        self.value = Lin(hidden, 1)
        self.advantage = Lin(hidden, n_actions)

    def forward(self, board: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        b = self.conv(board)
        b = b.flatten(1)
        b = self.board_fc(b)
        a = self.aux_fc(aux)
        h = torch.relu(self.trunk_lin(torch.cat([b, a], dim=1)))
        v = self.value(h)
        adv = self.advantage(h)
        return v + (adv - adv.mean(dim=1, keepdim=True))

    def reset_noise(self):
        """Re-amostra ruído em todas as NoisyLinear (chamar antes de cada passo)."""
        if not self.noisy:
            return
        for m in (self.trunk_lin, self.value, self.advantage):
            m.reset_noise()

    def eval_noise(self):
        """Zera o ruído (uso em avaliação gulosa)."""
        if not self.noisy:
            return
        for m in (self.trunk_lin, self.value, self.advantage):
            m.eval_noise()
