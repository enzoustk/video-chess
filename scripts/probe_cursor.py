"""Utilitário de engenharia reversa do input do Atari Video Chess.

Materializa e valida empiricamente todos os achados de RAM feitos no projeto,
com base no disassembly de Oscar Toledo G. (nanochess.org/video_chess.html):

    ram_D4 (byte 84)  -> square de ORIGEM do lance corrente (também é o cursor
                          livre em modo cursor / F3=0)
    ram_D5 (byte 85)  -> square de DESTINO do lance corrente (cursor em modo
                          seleção / F3=1)
    ram_DA (byte 90)  -> "current board score (incremental)" — NÃO é cursor
    ram_F3 (byte 115) -> máquina de estados:
                          $00 = cursor movendo
                          $01 = peça selecionada (origem travada)
                          $80 = peça movendo (animação do lance)
                          $c0 = engine buscando resposta
    ram_F5 (byte 117) -> validade do (origem, destino) corrente:
                          $00  = lance LEGAL
                          $FF  = lance ILEGAL

Encoding do cursor (K=3):  ``byte = 3 + 8·(7 - rank) + file``
  UP    -> byte -8 (rank +1)   RIGHT -> byte +1
  DOWN  -> byte +8 (rank -1)   LEFT  -> byte -1
  wrap: byte percorre 3..59 em passos de 8; após 59 volta a 3.

Debouncer: sob frameskip=4, ~1 movimento de cursor a cada 4 taps (o TIA amostra
o input a ~33 frames de repetição).

Uso:
    python -m scripts.probe_cursor
    python -m scripts.probe_cursor --sweep-legal   # varre destinos legais do
                                                   # peão em e2 usando F5
"""
from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np

try:
    import ale_py
    gym.register_envs(ale_py)
except Exception:
    pass

K = 3  # constante de encoding do cursor (byte = K + 8*(7-rank) + file)


def byte_to_rf(v):
    d = v - K
    return (7 - d // 8, d % 8)


def rf_to_byte(rank, file):
    return K + 8 * (7 - rank) + file


def make_raw():
    return gym.make("ALE/VideoChess-v5", obs_type="ram", frameskip=4,
                    repeat_action_probability=0.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep-legal", action="store_true",
                   help="após selecionar e2, varre destinos e imprime F5 (0=legal, 255=ilegal)")
    args = p.parse_args()

    env = make_raw()
    ale = env.unwrapped.ale
    env.reset(seed=0)
    for _ in range(40):
        env.step(0)   # settle

    def read():
        r = ale.getRAM()
        return (int(r[84]), int(r[85]), int(r[115]), int(r[117]))

    print(f"cursor inicial byte84={ale.getRAM()[84]}  ->  {byte_to_rf(int(ale.getRAM()[84]))}")
    print(f"encoding K={K}: byte = {K} + 8·(7-rank) + file")

    def move_one(d, watch=84):
        last = int(ale.getRAM()[watch])
        env.step(d)
        for _ in range(20):
            if int(ale.getRAM()[watch]) != last:
                for _ in range(8):
                    env.step(0)
                return True
            env.step(0)
        return False

    # navega para e2 (rank=1, file=4)
    print("\n[nav] navegando cursor (byte 84) para e2...")
    for _ in range(2): move_one(5, 84)   # DOWN
    for _ in range(4): move_one(3, 84)   # RIGHT
    v84, v85, f3, f5 = read()
    print(f"  cursor em byte84={v84} -> {byte_to_rf(v84)}  (esperado e2=(1,4) byte=55)")

    # FIRE seleciona
    print("\n[fire1] FIRE longo (~32 frames) para selecionar...")
    for _ in range(8): env.step(1)
    for _ in range(50): env.step(0)
    v84, v85, f3, f5 = read()
    print(f"  F3(115)={f3} (esperado 1=selected) | tgt(85)={v85} | F5(117)={f5}")

    if args.sweep_legal:
        # varre destinos em cruz (UP/DOWN/LEFT/RIGHT) e imprime F5 por casa
        print("\n[sweep] varrendo destinos e coletando F5 (0=legal):")
        for d_name, d in [("UP", 2), ("DOWN", 5), ("LEFT", 4), ("RIGHT", 3)]:
            env.reset(seed=0)
            for _ in range(40): env.step(0)
            for _ in range(2): move_one(5, 84)
            for _ in range(4): move_one(3, 84)
            for _ in range(8): env.step(1)
            for _ in range(50): env.step(0)
            seq = []
            for _ in range(7):
                if not move_one(d, 85): break
                for _ in range(6): env.step(0)
                _, v85, _, f5 = read()
                seq.append((v85, byte_to_rf(v85), f5))
            print(f"  {d_name}: {seq}")
        print("  → F5=0 destinos são os LANCES LEGAIS do engine para a peça selecionada")
    env.close()


if __name__ == "__main__":
    main()
