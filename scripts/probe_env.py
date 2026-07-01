"""Utilitário de inspeção do ambiente Video Chess (usado na engenharia reversa
do mapeamento RAM -> tabuleiro). Imprime espaços, RAM inicial, o tabuleiro
decodificado e o balanço material.

Uso:
    python -m scripts.probe_env
"""
from __future__ import annotations

import numpy as np

from chess_rl.board import decode_board, material_balance, render_ascii
from chess_rl.env import ENV_ID, make_env


def main():
    env = make_env(seed=0)
    print("Ambiente:", ENV_ID)
    print("Espaço de observação (RAM):", env.observation_space)
    print("Espaço de ações:", env.action_space)
    print("Ações:", env.unwrapped.get_action_meanings())

    ram, info = env.reset(seed=0)
    print("\nRAM inicial (8x16):")
    print(np.asarray(ram).reshape(8, 16))
    print("\nTabuleiro decodificado (maiúsc.=brancas):")
    print(render_ascii(ram))
    print("Balanço material inicial:", material_balance(ram))

    # rollout aleatório curto só para conferir dinâmica de recompensa modelada
    rng = np.random.default_rng(0)
    total = 0.0
    for _ in range(2000):
        a = env.action_space.sample()
        ram, r, term, trunc, info = env.step(a)
        total += r
        if term or trunc:
            break
    print("\nRecompensa modelada acumulada (2000 passos aleatórios):", round(total, 3))
    print("Balanço material final:", info.get("material_balance"))
    env.close()


if __name__ == "__main__":
    main()
