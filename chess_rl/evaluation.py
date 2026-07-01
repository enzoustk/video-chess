"""Heurística própria de avaliação de posição (estática), no estilo do chess-bot.

Avalia a posição decodificada da RAM combinando **material** com **piece-square
tables (PST)** — a mesma ideia de avaliação posicional usada por engines
clássicos (negamax + PST). É 100% Python/numpy, sem busca e sem subprocesso, de
modo que pode ser chamada a cada passo do ambiente sem custo relevante.

A avaliação é usada como **função de potencial** Φ(s) para *reward shaping*
baseado em potencial: ``r = γ·Φ(s') − Φ(s)`` (Ng et al., 1999), que é invariante
à política ótima (não introduz a degeneração de "mexer só pelo bônus").

Valor sempre na perspectiva das **brancas** (positivo = bom para as brancas);
``position_value(ram, agent_white=False)`` inverte o sinal para o agente preto.
"""
from __future__ import annotations

import numpy as np

from .board import BLACK_CODES, WHITE_CODES

# material em centipeões
MAT_CP = {"P": 100, "N": 320, "B": 330, "R": 500, "Q": 900, "K": 0}

# PST (Michniewski, "Simplified Evaluation Function"), centipeões, meio-jogo.
# Cada tabela está na ordem publicada: índice 0 = a8 (rank 8), ..., 63 = h1.
_PST_PUB = {
    "P": [
        0, 0, 0, 0, 0, 0, 0, 0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
        5, 5, 10, 25, 25, 10, 5, 5,
        0, 0, 0, 20, 20, 0, 0, 0,
        5, -5, -10, 0, 0, -10, -5, 5,
        5, 10, 10, -20, -20, 10, 10, 5,
        0, 0, 0, 0, 0, 0, 0, 0,
    ],
    "N": [
        -50, -40, -30, -30, -30, -30, -40, -50,
        -40, -20, 0, 0, 0, 0, -20, -40,
        -30, 0, 10, 15, 15, 10, 0, -30,
        -30, 5, 15, 20, 20, 15, 5, -30,
        -30, 0, 15, 20, 20, 15, 0, -30,
        -30, 5, 10, 15, 15, 10, 5, -30,
        -40, -20, 0, 5, 5, 0, -20, -40,
        -50, -40, -30, -30, -30, -30, -40, -50,
    ],
    "B": [
        -20, -10, -10, -10, -10, -10, -10, -20,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -10, 0, 5, 10, 10, 5, 0, -10,
        -10, 5, 5, 10, 10, 5, 5, -10,
        -10, 0, 10, 10, 10, 10, 0, -10,
        -10, 10, 10, 10, 10, 10, 10, -10,
        -10, 5, 0, 0, 0, 0, 5, -10,
        -20, -10, -10, -10, -10, -10, -10, -20,
    ],
    "R": [
        0, 0, 0, 0, 0, 0, 0, 0,
        5, 10, 10, 10, 10, 10, 10, 5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        0, 0, 0, 5, 5, 0, 0, 0,
    ],
    "Q": [
        -20, -10, -10, -5, -5, -10, -10, -20,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -10, 0, 5, 5, 5, 5, 0, -10,
        -5, 0, 5, 5, 5, 5, 0, -5,
        0, 0, 5, 5, 5, 5, 0, -5,
        -10, 5, 5, 5, 5, 5, 0, -10,
        -10, 0, 5, 0, 0, 0, 0, -10,
        -20, -10, -10, -5, -5, -10, -10, -20,
    ],
    "K": [  # meio-jogo: incentiva o roque/segurança do rei
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -20, -30, -30, -40, -40, -30, -30, -20,
        -10, -20, -20, -20, -20, -20, -20, -10,
        20, 20, 0, 0, 0, 0, 20, 20,
        20, 30, 10, 0, 0, 10, 30, 20,
    ],
}
_PST = {k: np.asarray(v, dtype=np.float32) for k, v in _PST_PUB.items()}

# Mapeia o índice linear do nosso tabuleiro (idx = rank*8 + file, rank 0 = base/
# brancas) para o índice da tabela publicada (rank 8 primeiro):
#   brancas: pub = (7 - rank)*8 + file
#   pretas : pub = rank*8 + file        (espelho vertical da tabela)
_WHITE_SQ = np.array([(7 - (i // 8)) * 8 + (i % 8) for i in range(64)], dtype=np.int64)
_BLACK_SQ = np.arange(64, dtype=np.int64)


def position_value(ram: np.ndarray, agent_white: bool = True) -> float:
    """Avaliação estática (material + PST) em **peões**, na perspectiva do agente."""
    b = ram[:64].astype(int)
    total = 0.0
    for idx in range(64):
        v = b[idx]
        if v == 0:
            continue
        if v in WHITE_CODES:
            p = WHITE_CODES[v]
            total += MAT_CP[p] + float(_PST[p][_WHITE_SQ[idx]])
        elif v in BLACK_CODES:
            p = BLACK_CODES[v]
            total -= MAT_CP[p] + float(_PST[p][_BLACK_SQ[idx]])
    total /= 100.0  # centipeões -> peões
    return total if agent_white else -total
