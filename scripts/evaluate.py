"""Avaliação de um agente treinado (política gulosa) com gravação opcional de vídeo.

Uso:
    python -m scripts.evaluate --run demo --episodes 5
    python -m scripts.evaluate --run demo --episodes 2 --video
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from chess_rl.agent import DQNAgent, get_device
from chess_rl.board import render_ascii
from chess_rl.config import Config
from chess_rl.env import make_env

RESULTS = Path(__file__).resolve().parent.parent / "results"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="demo")
    p.add_argument("--ckpt", default="best.pt")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--device", default=None)
    p.add_argument("--video", action="store_true", help="grava mp4 em results/<run>/video/")
    p.add_argument("--show-board", action="store_true", help="imprime tabuleiro final")
    p.add_argument("--random", action="store_true",
                   help="usa política aleatória (baseline, ignora o checkpoint)")
    args = p.parse_args()

    run = RESULTS / args.run
    cfg_dict = json.loads((run / "config.json").read_text())
    cfg = Config(**{k: v for k, v in cfg_dict.items() if k in Config().to_dict()})
    device = get_device(args.device)

    render_mode = "rgb_array" if args.video else None
    env = make_env(seed=cfg.seed, render_mode=render_mode,
                   reward_mode=cfg.reward_mode, material_scale=cfg.material_scale,
                   eval_scale=cfg.eval_scale, shaping_gamma=cfg.shaping_gamma,
                   reward_clip=cfg.reward_clip,
                   win_bonus=cfg.win_bonus, step_penalty=cfg.step_penalty,
                   move_bonus=cfg.move_bonus, agent_white=cfg.agent_white,
                   max_episode_steps=cfg.max_episode_steps,
                   frameskip=cfg.frameskip, sticky_prob=cfg.sticky_prob)
    if args.video:
        (run / "video").mkdir(exist_ok=True)
        env = gym.wrappers.RecordVideo(env, str(run / "video"),
                                       episode_trigger=lambda e: True,
                                       name_prefix="video_chess")

    agent = DQNAgent(env.action_space.n, cfg, device)
    if not args.random:
        agent.load(run / args.ckpt, map_location=device)
    else:
        print("[baseline] política ALEATÓRIA (checkpoint ignorado)")

    rewards, materials, lengths = [], [], []
    for ep in range(args.episodes):
        ram, info = env.reset(seed=20_000 + ep)
        done = False
        ep_r, steps = 0.0, 0
        while not done:
            if args.random:
                a = env.action_space.sample()
            else:
                a = agent.act(ram, step=0, greedy=True)
            ram, r, term, trunc, info = env.step(a)
            ep_r += r
            steps += 1
            done = term or trunc
        rewards.append(ep_r)
        materials.append(info.get("material_balance", 0.0))
        lengths.append(steps)
        print(f"ep {ep}: reward={ep_r:.2f} material={info.get('material_balance',0):+.1f} len={steps}")
        if args.show_board:
            print(render_ascii(ram), "\n")
    env.close()

    print("\n=== Resumo da avaliação ===")
    print(f"recompensa média:  {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"material médio:     {np.mean(materials):+.2f}")
    print(f"duração média:      {np.mean(lengths):.0f} passos")


if __name__ == "__main__":
    main()
