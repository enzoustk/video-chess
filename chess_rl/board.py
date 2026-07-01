"""Decodificação do tabuleiro a partir da RAM do Atari Video Chess.

A RAM do Video Chess (128 bytes) codifica o tabuleiro 8x8 **diretamente** nos
bytes 0..63 — um byte por casa, em ordem row-major (rank 0 = base, brancas).
O mapeamento abaixo foi descoberto por engenharia reversa da posição inicial
(ver ``scripts/probe_env.py``); a posição inicial de xadrez é inconfundível:

    rank 0 (bytes  0..7):   5  4  3  2  1  3  4  5    -> T C B D R B C T (brancas)
    rank 1 (bytes  8..15): 70 70 70 70 70 70 70 70    -> peões brancos
    ranks 2..5:             0 (casas vazias)
    rank 6 (bytes 48..55): 142 ...                     -> peões pretos
    rank 7 (bytes 56..63): 13 12 11 10  9 11 12 13    -> torres/.../rei pretos

Codificação:
    vazio   = 0
    Brancas: Rei=1, Dama=2, Bispo=3, Cavalo=4, Torre=5, Peão=70
    Pretas:  Rei=9, Dama=10, Bispo=11, Cavalo=12, Torre=13, Peão=142
"""
from __future__ import annotations

import numpy as np

# Decodificação CORRIGIDA via disassembly em nanochess.org/video_chess.html:
# "The contents of each square is located in the bits 3-0 of the byte, while
#  the upper bits 7-4 are used to save other data in a clever way."
# ou seja: tipo da peça = byte & 0x0F. Os bits altos são reusados (cursor,
# flags de roque, en passant, animação) e mudam durante o jogo — exigir o byte
# completo (ex.: 70 para peão branco) faz peças "sumirem" do decoder quando
# os bits altos mudam, gerando recompensa espúria.
WHITE_CODES = {1: "K", 2: "Q", 3: "B", 4: "N", 5: "R", 6: "P"}
BLACK_CODES = {9: "K", 10: "Q", 11: "B", 12: "N", 13: "R", 14: "P"}
PIECE_MASK = 0x0F  # mascara que isola o nibble baixo com o tipo da peça

# valor material padrão de xadrez (rei = 0; perda do rei é tratada pela
# recompensa nativa de fim de jogo).
PIECE_VALUE = {"K": 0.0, "Q": 9.0, "R": 5.0, "B": 3.0, "N": 3.0, "P": 1.0}

# índice de plano para o one-hot (0..5 brancas, 6..11 pretas)
PLANE_INDEX = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5,
               9: 6, 10: 7, 11: 8, 12: 9, 13: 10, 14: 11}
N_PLANES = 12

# valor de cada plano (em pontos) e sinal (brancas +, pretas -)
_PLANE_VALUE = np.array([0., 9., 3., 3., 5., 1.,
                         0., 9., 3., 3., 5., 1.], dtype=np.float32)
_PLANE_SIGN = np.array([1.] * 6 + [-1.] * 6, dtype=np.float32)
_PLANE_SIGNED_VALUE = _PLANE_VALUE * _PLANE_SIGN  # (12,)

# lookup vetorizado: byte (0..255) -> plano (0..11) ou -1 (vazio/desconhecido).
# Construído sobre o NIBBLE BAIXO de cada byte (& 0x0F), porque os bits 4-7 são
# reaproveitados pelo Video Chess para outras finalidades durante o jogo.
BYTE_TO_PLANE = np.full(256, -1, dtype=np.int64)
for _b in range(256):
    _nib = _b & PIECE_MASK
    if _nib in PLANE_INDEX:
        BYTE_TO_PLANE[_b] = PLANE_INDEX[_nib]


def decode_board(ram: np.ndarray) -> np.ndarray:
    """Retorna uma matriz 8x8 de strings legíveis (maiúsculas = brancas)."""
    cells = ram[:64].reshape(8, 8)
    out = np.full((8, 8), ".", dtype="<U1")
    for r in range(8):
        for c in range(8):
            v = int(cells[r, c]) & PIECE_MASK
            if v in WHITE_CODES:
                out[r, c] = WHITE_CODES[v]
            elif v in BLACK_CODES:
                out[r, c] = BLACK_CODES[v].lower()
    return out


def material_balance(ram: np.ndarray) -> float:
    """Balanço material (brancas - pretas) em pontos de xadrez."""
    planes = BYTE_TO_PLANE[ram[:64].astype(np.int64)]  # (64,) com -1 nas vazias
    valid = planes >= 0
    if not valid.any():
        return 0.0
    return float(_PLANE_SIGNED_VALUE[planes[valid]].sum())


def material_counts(ram: np.ndarray) -> tuple[float, float]:
    """(material_brancas, material_pretas) em pontos."""
    planes = BYTE_TO_PLANE[ram[:64].astype(np.int64)]
    valid = planes >= 0
    p = planes[valid]
    white = float(_PLANE_VALUE[p][p < 6].sum())
    black = float(_PLANE_VALUE[p][p >= 6].sum())
    return white, black


def render_ascii(ram: np.ndarray) -> str:
    """Tabuleiro em ASCII (rank 7 no topo, como visto pelas pretas em cima)."""
    b = decode_board(ram)
    lines = []
    for r in range(7, -1, -1):
        lines.append(" ".join(b[r, c] for c in range(8)))
    return "\n".join(lines)
