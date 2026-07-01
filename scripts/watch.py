"""Assistir o agente treinado jogando Video Chess.

Abre uma janela em tempo real (render_mode="human") mostrando o agente (brancas)
contra a IA do Atari (pretas). Use ``--ascii`` para imprimir o tabuleiro no
terminal em vez de abrir janela (útil em servidores sem display).

Uso:
    python -m scripts.watch --run dqn_move --ckpt best.pt --episodes 1 --fps 30
    python -m scripts.watch --run dqn_videochess --ascii
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from chess_rl.agent import DQNAgent, get_device
from chess_rl.board import material_balance, render_ascii
from chess_rl.config import Config
from chess_rl.env import make_env

RESULTS = Path(__file__).resolve().parent.parent / "results"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="dqn_move")
    p.add_argument("--ckpt", default="best.pt")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--device", default=None)
    p.add_argument("--epsilon", type=float, default=0.0,
                   help="ruído na política (0 = totalmente gulosa)")
    p.add_argument("--ascii", action="store_true",
                   help="imprime o tabuleiro no terminal em vez de abrir janela")
    args = p.parse_args()

    run = RESULTS / args.run
    cfg_dict = json.loads((run / "config.json").read_text())
    cfg = Config(**{k: v for k, v in cfg_dict.items() if k in Config().to_dict()})
    device = get_device(args.device)

    render_mode = None if args.ascii else "human"
    env = make_env(seed=cfg.seed, render_mode=render_mode,
                   reward_mode=cfg.reward_mode, material_scale=cfg.material_scale,
                   eval_scale=cfg.eval_scale, shaping_gamma=cfg.shaping_gamma,
                   reward_clip=cfg.reward_clip,
                   win_bonus=cfg.win_bonus, step_penalty=cfg.step_penalty,
                   move_bonus=cfg.move_bonus, agent_white=cfg.agent_white,
                   max_episode_steps=cfg.max_episode_steps,
                   frameskip=cfg.frameskip, sticky_prob=cfg.sticky_prob)

    agent = DQNAgent(env.action_space.n, cfg, device)
    agent.load(run / args.ckpt, map_location=device)
    delay = 1.0 / args.fps if args.fps > 0 else 0.0
    rng = np.random.default_rng(0)

    for ep in range(args.episodes):
        ram, info = env.reset(seed=40_000 + ep)
        done, steps, moves = False, 0, 0
        last_board = np.asarray(ram)[:64].copy()
        while not done:
            if args.epsilon > 0 and rng.random() < args.epsilon:
                a = int(rng.integers(0, env.action_space.n))
            else:
                a = agent.act(ram, step=0, greedy=True)
            ram, r, term, trunc, info = env.step(a)
            steps += 1
            board = np.asarray(ram)[:64]
            if not np.array_equal(board, last_board):
                moves += 1
                last_board = board.copy()
                if args.ascii:
                    print(f"\n--- lance {moves} (passo {steps}) | "
                          f"material {material_balance(ram):+.0f} ---")
                    print(render_ascii(ram))
            done = term or trunc
            if delay:
                time.sleep(delay)
        print(f"[ep {ep}] passos={steps} lances={moves} "
              f"material_final={info.get('material_balance',0):+.0f}")
    env.close()


if __name__ == "__main__":
    main()
