"""Wrapper *chess-level* do Video Chess.

Substitui o espaço de ações do Atari (`Discrete(10)` de cursor) por um espaço
de **lances de xadrez** (`Discrete(4096)` = 64 origens × 64 destinos). Usa
``scripted_moves.execute_move`` para traduzir cada lance no cursor real do
Atari, aproveitando a IA de xadrez do console como oponente.

Isso pula completamente o aprendizado da meta-mecânica do cursor: o RL agora
**aprende xadrez de verdade**, não navegação de cursor. Todas as demais peças
do pipeline (recompensa material/PST, Dueling Double DQN, NoisyNets etc.)
continuam funcionando; muda apenas o espaço de ações e o passo de execução.

Action masking: em cada estado retornamos ``info["action_mask"]`` (bool[4096])
com apenas os lances legais para as pretas ativos. O agente deve mascarar as
ações inválidas antes do ``argmax``.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

try:
    import ale_py
    gym.register_envs(ale_py)
except Exception:
    pass

import chess

from .board import material_balance
from .evaluation import position_value
from .scripted_moves import (
    board_to_python_chess,
    execute_move,
    wait_engine_response,
)

ENV_ID = "ALE/VideoChess-v5"
N_SQUARES = 64
N_ACTIONS = N_SQUARES * N_SQUARES   # 4096


def move_to_action(mv: chess.Move) -> int:
    return mv.from_square * 64 + mv.to_square


def action_to_squares(a: int) -> tuple[int, int]:
    return a // 64, a % 64


class VideoChessMoveEnv(gym.Env):
    """Env com action space = lances de xadrez, executados via cursor scriptado.

    Reward por lance = escala · (Φ(s') − Φ(s)), onde Φ é a heurística material+PST
    da perspectiva do agente (PRETAS). Um lance ilegal (não na máscara) recebe
    penalidade fixa e o estado não muda.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, max_moves: int = 40, eval_scale: float = 0.1,
                 illegal_penalty: float = -0.5, win_bonus: float = 1.0,
                 max_engine_wait: int = 4000, seed: int | None = None,
                 reward_mode: str = "eval"):
        """reward_mode: 'eval' = Δ Φ (material+PST heuristic);
                        'material' = Δ balanço material puro (menos exploitável)."""
        self.reward_mode = reward_mode
        super().__init__()
        self.env = gym.make(ENV_ID, obs_type="ram", frameskip=4,
                            repeat_action_probability=0.0)
        self.observation_space = gym.spaces.Box(0, 255, (128,), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(N_ACTIONS)
        self.max_moves = max_moves
        self.eval_scale = eval_scale
        self.illegal_penalty = illegal_penalty
        self.win_bonus = win_bonus
        self.max_engine_wait = max_engine_wait
        self._moves_done = 0
        self._prev_phi = 0.0
        self._prev_mat_pov = 0.0   # material da perspectiva do agente (pretas)
        self._seed = seed

    # ---------- utilidades ----------
    def _ale(self):
        return self.env.unwrapped.ale

    def _legal_moves(self) -> list[chess.Move]:
        board = board_to_python_chess(self._ale().getRAM(), black_to_move=True)
        return list(board.legal_moves)

    def _action_mask(self, moves: list[chess.Move]) -> np.ndarray:
        m = np.zeros(N_ACTIONS, dtype=bool)
        for mv in moves:
            m[move_to_action(mv)] = True
        return m

    # ---------- gym API ----------
    def reset(self, *, seed: int | None = None, options=None):
        seed = seed if seed is not None else self._seed
        obs, info = self.env.reset(seed=seed)
        for _ in range(30):
            self.env.step(0)  # settle intro
        ram = self._ale().getRAM()
        self._prev_phi = position_value(ram, agent_white=False)
        self._prev_mat_pov = -material_balance(ram)   # black perspective
        self._moves_done = 0
        moves = self._legal_moves()
        info = {"action_mask": self._action_mask(moves),
                "legal_move_count": len(moves),
                "phi": self._prev_phi,
                "material": material_balance(ram)}
        return ram.copy(), info

    def _game_result(self, chess_board: "chess.Board") -> tuple[str, str]:
        """Detecta o motivo de fim + o vencedor. Chamado quando não há legais."""
        # tabuleiro python-chess reconstruído; usamos-o pra classificar o fim
        if chess_board.is_checkmate():
            winner = "white" if chess_board.turn == chess.BLACK else "black"
            return "checkmate", winner
        if chess_board.is_stalemate():
            return "stalemate", "draw"
        if chess_board.is_insufficient_material():
            return "insufficient_material", "draw"
        if chess_board.can_claim_fifty_moves():
            return "fifty_moves", "draw"
        if chess_board.can_claim_threefold_repetition():
            return "repetition", "draw"
        return "no_legal_moves", "unknown"

    def step(self, action: int):
        ale = self._ale()
        pre = ale.getRAM()[:64].copy()
        legals = self._legal_moves()
        legal_actions = {move_to_action(mv) for mv in legals}
        # ação ilegal: penalidade, sem executar
        if action not in legal_actions:
            reward = self.illegal_penalty
            ram = ale.getRAM()
            info = {"action_mask": self._action_mask(legals),
                    "legal_move_count": len(legals), "illegal": True,
                    "phi": position_value(ram, agent_white=False),
                    "material": material_balance(ram)}
            return ram.copy(), reward, False, False, info
        # executa lance
        src, dst = action_to_squares(action)
        exec_ok = execute_move(self.env, src, dst)
        engine_responded = False
        if exec_ok:
            engine_responded = wait_engine_response(self.env, pre,
                                                    max_steps=self.max_engine_wait)
        ram = ale.getRAM()
        phi = position_value(ram, agent_white=False)
        mat_pov = -material_balance(ram)   # black perspective
        if self.reward_mode == "material":
            delta = mat_pov - self._prev_mat_pov
        else:  # "eval" (material + PST)
            delta = phi - self._prev_phi
        self._prev_phi = phi
        self._prev_mat_pov = mat_pov
        reward = float(self.eval_scale * delta)
        self._moves_done += 1
        # próximo conjunto de legais
        try:
            next_moves = self._legal_moves()
        except Exception:
            next_moves = []
        term = len(next_moves) == 0    # sem movimentos = mate/pat/etc.
        trunc = (self.max_moves is not None) and (self._moves_done >= self.max_moves)
        end_reason = None; winner = None
        if term or trunc:
            try:
                cb = board_to_python_chess(ram, black_to_move=True)
                # tenta o outro turno também (caso mate esteja no turno das brancas)
                cb.turn = chess.BLACK if len(next_moves) > 0 else cb.turn
                end_reason, winner = self._game_result(cb)
            except Exception:
                end_reason, winner = ("error", "unknown")
            if trunc and not term:
                end_reason = "truncated"; winner = "unknown"
        info = {"action_mask": self._action_mask(next_moves),
                "legal_move_count": len(next_moves),
                "exec_ok": exec_ok, "engine_responded": engine_responded,
                "phi": phi, "material": material_balance(ram),
                "moves_done": self._moves_done,
                "end_reason": end_reason, "winner": winner}
        return ram.copy(), reward, term, trunc, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
