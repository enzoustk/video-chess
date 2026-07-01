"""Jogar Video Chess você mesmo (teclado), contra a IA do Atari.

Permite experimentar a mesma mecânica de cursor que o agente precisa aprender:
mover o cursor até uma peça, FIRE para selecionar, mover até o destino, FIRE
para soltar. É um ótimo jeito de entender por que o problema é difícil para RL.

Controles:
    Setas      -> mover o cursor (UP/DOWN/LEFT/RIGHT)
    W/E/A/D    -> diagonais (UP-LEFT, UP-RIGHT, DOWN-LEFT, DOWN-RIGHT)
    ESPAÇO     -> FIRE (selecionar / soltar peça)
    ESC        -> sair

Uso:
    python -m scripts.play_human
"""
from __future__ import annotations

import argparse

import gymnasium as gym

try:
    import ale_py
    gym.register_envs(ale_py)
except Exception:
    pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--frameskip", type=int, default=2)
    args = p.parse_args()

    try:
        import pygame
        from gymnasium.utils.play import play
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Este modo precisa de pygame e de um display gráfico. "
            f"Erro: {e}\nInstale com: pip install pygame"
        )

    keys_to_action = {
        (pygame.K_SPACE,): 1,                          # FIRE
        (pygame.K_UP,): 2,
        (pygame.K_RIGHT,): 3,
        (pygame.K_LEFT,): 4,
        (pygame.K_DOWN,): 5,
        (pygame.K_e,): 6,                              # UP-RIGHT
        (pygame.K_w,): 7,                              # UP-LEFT
        (pygame.K_d,): 8,                              # DOWN-RIGHT
        (pygame.K_a,): 9,                              # DOWN-LEFT
        (pygame.K_UP, pygame.K_RIGHT): 6,
        (pygame.K_UP, pygame.K_LEFT): 7,
        (pygame.K_DOWN, pygame.K_RIGHT): 8,
        (pygame.K_DOWN, pygame.K_LEFT): 9,
    }

    env = gym.make("ALE/VideoChess-v5", render_mode="rgb_array",
                   frameskip=args.frameskip, repeat_action_probability=0.0)
    print(__doc__)
    play(env, keys_to_action=keys_to_action, noop=0, fps=args.fps)


if __name__ == "__main__":
    main()
