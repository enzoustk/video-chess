"""Transformação RAM -> representação de estado para a rede neural.

O estado é dividido em duas partes que respeitam a *natureza* do problema:

* **tabuleiro** -> tensor 12x8x8 (one-hot de 12 tipos de peça), processado por
  uma CNN (estrutura espacial 2D, equivariante à posição das peças);
* **auxiliar** -> bytes 64..127 da RAM normalizados (posição do cursor,
  indicador de turno, contadores de animação/UI), processado por um MLP.

A codificação é vetorizada para um *batch* inteiro, o que mantém o laço de
treino rápido mesmo armazenando apenas a RAM crua (128 bytes) no replay buffer.
"""
from __future__ import annotations

import numpy as np

from .board import BYTE_TO_PLANE, N_PLANES

AUX_START = 64
AUX_LEN = 64  # bytes 64..127
BOARD_SHAPE = (N_PLANES, 8, 8)


def encode_batch(ram_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(N,128) uint8`` -> ``(board (N,12,8,8) float32, aux (N,64) float32)``."""
    ram_batch = np.asarray(ram_batch)
    if ram_batch.ndim == 1:
        ram_batch = ram_batch[None, :]
    n = ram_batch.shape[0]

    board_bytes = ram_batch[:, :64].astype(np.int64)        # (N,64)
    planes_idx = BYTE_TO_PLANE[board_bytes]                  # (N,64), -1 vazia
    board = np.zeros((n, N_PLANES, 64), dtype=np.float32)
    rows, sqs = np.where(planes_idx >= 0)
    board[rows, planes_idx[rows, sqs], sqs] = 1.0
    board = board.reshape(n, N_PLANES, 8, 8)

    aux = ram_batch[:, AUX_START:AUX_START + AUX_LEN].astype(np.float32) / 255.0
    return board, aux
