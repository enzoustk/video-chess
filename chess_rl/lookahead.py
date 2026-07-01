"""Políticas com lookahead (busca em árvore) para o agente de xadrez.

Envolve a política gulosa de Q-values com uma busca *minimax* de profundidade
`depth`, usando `python-chess` para simular as continuações e a heurística
material+PST (ou o próprio Q-value) como função de avaliação.

Isso ataca o gargalo diagnóstico de "o agente só olha 1-ply à frente" ao
resolver posições táticas com o motor Atari (que planeja 5+ plies).

Uso típico (na `benchmark_chess.py` ou `watch_chess.py`):

    from chess_rl.lookahead import minimax_move
    action = minimax_move(ram, mask, depth=2, agent_black=True)
"""
from __future__ import annotations

import chess
import numpy as np

from .chess_env import action_to_squares, move_to_action
from .evaluation import MAT_CP, _PST
from .scripted_moves import board_to_python_chess

# valores em centipeões
_PIECE_VAL = {chess.PAWN: MAT_CP["P"], chess.KNIGHT: MAT_CP["N"],
              chess.BISHOP: MAT_CP["B"], chess.ROOK: MAT_CP["R"],
              chess.QUEEN: MAT_CP["Q"], chess.KING: MAT_CP["K"]}
_PIECE_LETTER = {chess.KING: "K", chess.QUEEN: "Q", chess.ROOK: "R",
                 chess.BISHOP: "B", chess.KNIGHT: "N", chess.PAWN: "P"}

# lookup do PST para pretas: espelha rank vertical (sq XOR 56).
_MIRROR = np.array([sq ^ 0x38 for sq in range(64)], dtype=np.int64)


def evaluate_pychess(board: chess.Board, agent_black: bool = True) -> float:
    """Avaliação estática (material + PST) na perspectiva do agente, em peões."""
    total = 0.0
    for sq, piece in board.piece_map().items():
        val = _PIECE_VAL[piece.piece_type]
        pst = _PST[_PIECE_LETTER[piece.piece_type]]
        # PST publicado indexa "a8..h1" (index 0 = a8 rank8). Nosso engine usa
        # linear rank*8+file. Precisamos: pub_index = (7 - rank) * 8 + file.
        if piece.color == chess.WHITE:
            r, f = sq // 8, sq % 8
            pst_v = float(pst[(7 - r) * 8 + f])
        else:
            r, f = sq // 8, sq % 8
            pst_v = float(pst[r * 8 + f])   # espelhado para pretas
        signed = val + pst_v
        if piece.color == chess.WHITE:
            total += signed
        else:
            total -= signed
    total /= 100.0   # centipeões → peões
    return total if not agent_black else -total


def minimax(board: chess.Board, depth: int, agent_black: bool,
            alpha: float = -1e9, beta: float = 1e9,
            our_turn: bool = True) -> float:
    """Minimax com poda alpha-beta. Retorna avaliação na perspectiva do agente.

    ``our_turn`` alterna: True = maximizamos; False = adversário minimiza.
    """
    if depth == 0 or not board.legal_moves:
        return evaluate_pychess(board, agent_black)
    if our_turn:
        best = -1e9
        for mv in board.legal_moves:
            board.push(mv)
            v = minimax(board, depth - 1, agent_black, alpha, beta, our_turn=False)
            board.pop()
            best = max(best, v)
            alpha = max(alpha, v)
            if alpha >= beta:
                break
        return best
    else:
        worst = 1e9
        for mv in board.legal_moves:
            board.push(mv)
            v = minimax(board, depth - 1, agent_black, alpha, beta, our_turn=True)
            board.pop()
            worst = min(worst, v)
            beta = min(beta, v)
            if alpha >= beta:
                break
        return worst


def minimax_move(ram: np.ndarray, action_mask: np.ndarray,
                 depth: int = 2, agent_black: bool = True) -> int:
    """Retorna a **ação Discrete(4096)** escolhida por minimax de profundidade `depth`.

    depth=2 significa: nosso lance → resposta do oponente → avaliar (2 plies).
    depth=1 apenas: nosso lance → avaliar (1 ply, ignora resposta).
    """
    board = board_to_python_chess(ram, black_to_move=agent_black)
    best_action = 0
    best_val = -1e9
    for mv in board.legal_moves:
        board.push(mv)
        # após nosso lance, é vez do oponente. Se depth>1, oponente joga também.
        if depth <= 1:
            v = evaluate_pychess(board, agent_black)
        else:
            v = minimax(board, depth - 1, agent_black, our_turn=False)
        board.pop()
        act = move_to_action(mv)
        if not action_mask[act]:
            continue
        if v > best_val:
            best_val = v
            best_action = act
    return best_action
