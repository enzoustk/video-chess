"""Criação do ambiente Video Chess e wrapper de modelagem de recompensa.

A recompensa nativa do Video Chess é *extremamente esparsa* (apenas +/- ao
fim da partida, que jogo aleatório nunca alcança). Para que o agente receba
sinal de aprendizado, aplicamos **reward shaping** baseado no balanço material
lido da RAM:

    r_t = clip(escala * Δ(material), -clip, +clip)   (captura/perda de peças)
        + bonus_vitoria * r_nativo                    (fim de jogo: vitória/derrota)
        + penalidade_passo                            (incentiva progredir)

Isso transforma um problema intratável (curva de aprendizado plana em zero) em
um problema com gradiente de recompensa denso e interpretável.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

try:  # registro dos ambientes ALE
    import ale_py
    gym.register_envs(ale_py)
except Exception:  # pragma: no cover
    pass

from .board import material_balance
from .evaluation import position_value

ENV_ID = "ALE/VideoChess-v5"


class VideoChessRewardShaping(gym.Wrapper):
    """Recompensa densa lida da RAM. Dois modos de modelagem (``reward_mode``):

    * ``"material"`` -> r = escala · Δ(balanço material)  (clipado);
    * ``"eval"``     -> *reward shaping* baseado em potencial com a heurística
      de avaliação posicional (material + PST):  r = escala · (γ·Φ(s') − Φ(s)).

    Em ambos: + win_bonus·r_nativo (fim de jogo) + step_penalty + move_bonus
    (por lance legal).
    """

    def __init__(self, env, reward_mode="material", material_scale=0.1,
                 eval_scale=0.1, shaping_gamma=1.0, reward_clip=1.0,
                 win_bonus=1.0, step_penalty=0.0, move_bonus=0.0,
                 agent_white=True):
        super().__init__(env)
        self.reward_mode = reward_mode
        self.material_scale = material_scale
        self.eval_scale = eval_scale
        self.shaping_gamma = shaping_gamma
        self.reward_clip = reward_clip
        self.win_bonus = win_bonus
        self.step_penalty = step_penalty
        self.move_bonus = move_bonus  # bônus por lance legal (mudança no tabuleiro)
        self.agent_white = agent_white
        self.agent_sign = 1.0 if agent_white else -1.0
        self._prev_balance = 0.0
        self._prev_phi = 0.0
        self._prev_board = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_balance = material_balance(obs)
        self._prev_phi = position_value(obs, self.agent_white)
        self._prev_board = np.asarray(obs)[:64].copy()
        info["material_balance"] = self._prev_balance
        info["position_value"] = self._prev_phi
        return obs, info

    def step(self, action):
        obs, native_reward, terminated, truncated, info = self.env.step(action)
        balance = material_balance(obs)
        phi = position_value(obs, self.agent_white)

        board = np.asarray(obs)[:64]
        board_changed = not np.array_equal(board, self._prev_board)
        self._prev_board = board.copy()

        if self.reward_mode == "eval":
            # shaping baseado em potencial: F = γ·Φ(s') − Φ(s)
            core = self.eval_scale * (self.shaping_gamma * phi - self._prev_phi)
        else:
            core = self.material_scale * self.agent_sign * (balance - self._prev_balance)

        self._prev_balance = balance
        self._prev_phi = phi

        shaped = core
        if self.reward_clip and self.reward_clip > 0:
            shaped = float(np.clip(shaped, -self.reward_clip, self.reward_clip))
        shaped += self.win_bonus * float(native_reward)
        shaped += self.step_penalty
        if board_changed:
            shaped += self.move_bonus

        info["material_balance"] = balance
        info["position_value"] = phi
        info["native_reward"] = float(native_reward)
        info["board_changed"] = board_changed
        info["shaped_reward"] = shaped
        return obs, shaped, terminated, truncated, info


def make_env(seed=0, render_mode=None, max_episode_steps=2000, frameskip=4,
             sticky_prob=0.0, reward_mode="material", material_scale=0.1,
             eval_scale=0.1, shaping_gamma=1.0, reward_clip=1.0,
             win_bonus=1.0, step_penalty=0.0, move_bonus=0.0, agent_white=True):
    """Cria o ambiente Video Chess (observação = RAM) já com shaping e TimeLimit."""
    env = gym.make(
        ENV_ID,
        obs_type="ram",
        render_mode=render_mode,
        frameskip=frameskip,
        repeat_action_probability=sticky_prob,
    )
    if max_episode_steps:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    env = VideoChessRewardShaping(
        env, reward_mode=reward_mode, material_scale=material_scale,
        eval_scale=eval_scale, shaping_gamma=shaping_gamma,
        reward_clip=reward_clip, win_bonus=win_bonus, step_penalty=step_penalty,
        move_bonus=move_bonus, agent_white=agent_white,
    )
    if seed is not None:
        env.reset(seed=seed)
    return env
