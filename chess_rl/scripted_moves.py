"""Execução scriptada de lances de xadrez no ambiente Atari Video Chess.

Baseado na engenharia reversa do disassembly (nanochess.org). O agente humano
padrão no ``mode=0`` do Video Chess joga com as **pretas** (o motor Atari joga
com as brancas). A codificação dos squares na RAM é **row-major direto**
(0..63), com rank 0 = rank 1 do xadrez, rank 7 = rank 8 (bytes 0-63 são o
tabuleiro; a máscara `& 0x0F` isola o tipo da peça).

RAM bytes chave (via `ale.getRAM()`):
    84 (ram_D4) — casa de origem / cursor livre em F3=0
    85 (ram_D5) — casa de destino / cursor em modo seleção F3=1
   115 (ram_F3) — máquina de estados
   117 (ram_F5) — validade do lance corrente (0 = legal, ≠0 = ilegal)

O joystick move o cursor por offsets ±1 (LEFT/RIGHT) e ±8 (UP/DOWN) sobre o
byte 84 (em F3=0) ou byte 85 (em F3=1), com wrap `& 0x3f`. Debouncer de ~33
frames — sob frameskip=4 usamos padrão *tap + poll* até o byte mudar.

FIRE precisa de ~32 frames de hold (rising edge detection); depois release para
próximo evento registrar.

API:
    execute_move(env, src_sq, dst_sq) -> bool
        Faz o lance src_sq → dst_sq (índices row-major 0..63).
        Retorna True se sequencia terminou (não garante que o motor aceitou;
        chame ``move_was_valid`` depois para conferir via F5).

    play_random_legal(env) -> chess.Move | None
        Decodifica o tabuleiro, pede um lance legal aleatório (python-chess),
        executa via execute_move e retorna o Move.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import chess  # python-chess
except ImportError:  # pragma: no cover
    chess = None

from .board import BLACK_CODES, PIECE_MASK, WHITE_CODES

# ações do minimal action set do VideoChess
NOOP, FIRE, UP, RIGHT, LEFT, DOWN = 0, 1, 2, 3, 4, 5


def _cursor_free(ale):
    return int(ale.getRAM()[84])


def _cursor_target(ale):
    return int(ale.getRAM()[85])


def _state(ale):
    return int(ale.getRAM()[115])


def _valid_flag(ale):
    return int(ale.getRAM()[117])


def _move_one(env, direction: int, watch: int, max_wait: int = 20) -> bool:
    """Aplica uma direção e aguarda ate o byte 'watch' mudar (single square)."""
    ale = env.unwrapped.ale
    last = int(ale.getRAM()[watch])
    for _ in range(max_wait):
        env.step(direction)
        if int(ale.getRAM()[watch]) != last:
            for _ in range(6):
                env.step(0)  # settle
            return True
        env.step(0)
    return False


def _fire(env, hold_steps: int = 8, settle: int = 40):
    """Pressa FIRE por ~32 frames (o mínimo detectável) e solta."""
    for _ in range(hold_steps):
        env.step(FIRE)
    for _ in range(settle):
        env.step(NOOP)


def _navigate(env, target_byte: int, watch: int, max_iters: int = 80) -> bool:
    """Move o cursor (byte 'watch') passo-a-passo até o valor 'target_byte'."""
    ale = env.unwrapped.ale
    for _ in range(max_iters):
        cur = int(ale.getRAM()[watch])
        if cur == target_byte:
            return True
        delta = target_byte - cur
        if abs(delta) >= 8:
            d = DOWN if delta > 0 else UP
        else:
            d = RIGHT if delta > 0 else LEFT
        if not _move_one(env, d, watch):
            return False
    return int(ale.getRAM()[watch]) == target_byte


def execute_move(env, src_sq: int, dst_sq: int) -> bool:
    """Executa um lance de xadrez usando FIRE mechanic. Índices row-major 0..63."""
    ale = env.unwrapped.ale
    # navega cursor livre até a origem
    if not _navigate(env, src_sq, watch=84):
        return False
    _fire(env)
    if _state(ale) != 1:  # não selecionou
        return False
    # navega cursor de destino
    if not _navigate(env, dst_sq, watch=85):
        return False
    _fire(env)
    return True


def move_was_valid(env) -> bool:
    """Após execute_move, checa se o engine aceitou (F5 == 0)."""
    return _valid_flag(env.unwrapped.ale) == 0


def wait_engine_response(env, pre_move_board: np.ndarray,
                         max_steps: int = 4000) -> bool:
    """Aguarda até o motor Atari (brancas) responder ao nosso lance.

    Passa ``pre_move_board`` = snapshot dos bytes 0-63 ANTES de chamar
    ``execute_move``. Retorna quando pelo menos uma peça BRANCA (low nibble
    1..6) mudou de casa, indicando que o engine já jogou. Faz um settle final
    para estabilizar animações.
    """
    ale = env.unwrapped.ale
    pre_lo = pre_move_board & 0x0F
    for _ in range(max_steps):
        env.step(NOOP)
        cur_lo = ale.getRAM()[:64] & 0x0F
        # peças brancas que mudaram (apareceram OU saíram vs. pre-move)
        white_changes = 0
        for sq in range(64):
            p_before = int(pre_lo[sq])
            p_after = int(cur_lo[sq])
            if p_before != p_after and (1 <= p_before <= 6 or 1 <= p_after <= 6):
                white_changes += 1
                if white_changes >= 2:  # brancas source + dest
                    for _ in range(30):
                        env.step(NOOP)  # settle
                    return True
    return False


def board_to_python_chess(ram: np.ndarray, black_to_move: bool = True) -> "chess.Board":
    """Converte a RAM decodificada para um chess.Board do python-chess."""
    if chess is None:
        raise ImportError("Precisa de python-chess: pip install chess")
    b = chess.Board.empty()
    for sq in range(64):
        v = int(ram[sq]) & PIECE_MASK
        if v == 0:
            continue
        if v in WHITE_CODES:
            piece_type_map = {"K": chess.KING, "Q": chess.QUEEN, "R": chess.ROOK,
                              "B": chess.BISHOP, "N": chess.KNIGHT, "P": chess.PAWN}
            b.set_piece_at(sq, chess.Piece(piece_type_map[WHITE_CODES[v]], chess.WHITE))
        elif v in BLACK_CODES:
            piece_type_map = {"K": chess.KING, "Q": chess.QUEEN, "R": chess.ROOK,
                              "B": chess.BISHOP, "N": chess.KNIGHT, "P": chess.PAWN}
            b.set_piece_at(sq, chess.Piece(piece_type_map[BLACK_CODES[v]], chess.BLACK))
    b.turn = chess.BLACK if black_to_move else chess.WHITE
    return b


def play_random_legal(env, rng: Optional[np.random.Generator] = None) -> "chess.Move | None":
    """Escolhe um lance legal ALEATÓRIO (para o time PRETO) e executa."""
    if chess is None:
        raise ImportError("Precisa de python-chess: pip install chess")
    if rng is None:
        rng = np.random.default_rng()
    ram = env.unwrapped.ale.getRAM()
    board = board_to_python_chess(ram, black_to_move=True)
    legals = list(board.legal_moves)
    if not legals:
        return None
    move = legals[rng.integers(0, len(legals))]
    ok = execute_move(env, move.from_square, move.to_square)
    return move if ok else None
