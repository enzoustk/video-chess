"""NoisyNets — exploração paramétrica (Fortunato et al., 2017).

Substitui a camada Linear padrão por NoisyLinear: o peso e o bias ganham
componentes "ruidosos" cujos fatores de escala (σ) são aprendidos por
gradiente. O ruído é amostrado por *forward pass*, fazendo a política
explorar de forma **estado-dependente** — superior ao ε-greedy clássico em
ambientes de recompensa esparsa (peça padrão do Rainbow DQN).

Cada NoisyLinear mantém:
    μ_W, σ_W, μ_b, σ_b   (parâmetros aprendidos)
    ε_W, ε_b             (ruído amostrado, fatorado por Factorized Gaussian)
    W = μ_W + σ_W ⊙ ε_W   (peso efetivo da pass corrente)

`reset_noise()` re-amostra ε. Em modo de avaliação (sem ruído), basta zerar ε
via `eval_noise()`.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        # parametros aprendidos
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        # buffers de ruido (nao aprendidos, mas resamplados)
        self.register_buffer("weight_eps", torch.empty(out_features, in_features))
        self.register_buffer("bias_eps", torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        bound = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)
        s = self.sigma_init / math.sqrt(self.in_features)
        self.weight_sigma.data.fill_(s)
        self.bias_sigma.data.fill_(s)

    @staticmethod
    def _scale_noise(size: int, device) -> torch.Tensor:
        x = torch.randn(size, device=device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        """Re-amostra o ruido factorized Gaussian."""
        device = self.weight_mu.device
        eps_in = self._scale_noise(self.in_features, device)
        eps_out = self._scale_noise(self.out_features, device)
        self.weight_eps.copy_(eps_out.unsqueeze(1) * eps_in.unsqueeze(0))
        self.bias_eps.copy_(eps_out)

    def eval_noise(self):
        """Zera o ruido (uso em avaliacao 'gulosa' sem exploracao)."""
        self.weight_eps.zero_()
        self.bias_eps.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight_mu + self.weight_sigma * self.weight_eps
        b = self.bias_mu + self.bias_sigma * self.bias_eps
        return F.linear(x, w, b)
