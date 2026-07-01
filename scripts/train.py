"""Treino do agente Double/Dueling DQN no Atari Video Chess.

Uso (exemplos):
    python -m scripts.train --run demo --total-steps 150000
    python -m scripts.train --run full --total-steps 1000000 --device mps

Gera em ``results/<run>/``:
    config.json     -> hiperparâmetros usados
    log.csv         -> métricas por episódio (recompensa, material, perda, eps)
    eval.csv        -> avaliações gulosas periódicas
    best.pt / last.pt -> checkpoints
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import deque
from pathlib import Path

import numpy as np

from chess_rl.agent import DQNAgent, get_device
from chess_rl.config import Config
from chess_rl.env import make_env
from chess_rl.replay import ReplayBuffer

RESULTS = Path(__file__).resolve().parent.parent / "results"


def evaluate(agent, cfg, episodes, seed_base=10_000):
    """Roda episódios gulosos (epsilon=0) e retorna métricas médias."""
    env = make_env(seed=seed_base, reward_mode=cfg.reward_mode,
                   material_scale=cfg.material_scale, eval_scale=cfg.eval_scale,
                   shaping_gamma=cfg.shaping_gamma,
                   reward_clip=cfg.reward_clip, win_bonus=cfg.win_bonus,
                   step_penalty=cfg.step_penalty, move_bonus=cfg.move_bonus,
                   agent_white=cfg.agent_white,
                   max_episode_steps=cfg.max_episode_steps,
                   frameskip=cfg.frameskip, sticky_prob=cfg.sticky_prob)
    rewards, materials, lengths = [], [], []
    for ep in range(episodes):
        ram, info = env.reset(seed=seed_base + ep)
        done = False
        ep_r, steps = 0.0, 0
        while not done:
            a = agent.act(ram, step=0, greedy=True)
            ram, r, term, trunc, info = env.step(a)
            ep_r += r
            steps += 1
            done = term or trunc
        rewards.append(ep_r)
        materials.append(info.get("material_balance", 0.0))
        lengths.append(steps)
    env.close()
    return (float(np.mean(rewards)), float(np.mean(materials)),
            float(np.mean(lengths)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="demo")
    p.add_argument("--total-steps", type=int, default=None)
    p.add_argument("--device", default=None, help="mps|cuda|cpu (auto se omitido)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eps-decay-steps", type=int, default=None)
    p.add_argument("--learning-starts", type=int, default=None)
    p.add_argument("--eval-freq", type=int, default=None)
    p.add_argument("--eval-episodes", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--material-scale", type=float, default=None)
    p.add_argument("--reward-mode", default=None, choices=["material", "eval"])
    p.add_argument("--eval-scale", type=float, default=None)
    p.add_argument("--move-bonus", type=float, default=None)
    p.add_argument("--step-penalty", type=float, default=None)
    p.add_argument("--use-noisy", action="store_true",
                   help="NoisyNets: substitui ε-greedy por ruído paramétrico")
    p.add_argument("--noisy-sigma", type=float, default=None,
                   help="σ inicial dos NoisyLinear (padrão 0.5)")
    args = p.parse_args()

    cfg = Config(seed=args.seed)
    if args.total_steps is not None:
        cfg.total_steps = args.total_steps
    if args.eps_decay_steps is not None:
        cfg.eps_decay_steps = args.eps_decay_steps
    if args.learning_starts is not None:
        cfg.learning_starts = args.learning_starts
    if args.eval_freq is not None:
        cfg.eval_freq = args.eval_freq
    if args.eval_episodes is not None:
        cfg.eval_episodes = args.eval_episodes
    if args.lr is not None:
        cfg.lr = args.lr
    if args.material_scale is not None:
        cfg.material_scale = args.material_scale
    if args.reward_mode is not None:
        cfg.reward_mode = args.reward_mode
    if args.eval_scale is not None:
        cfg.eval_scale = args.eval_scale
    if args.move_bonus is not None:
        cfg.move_bonus = args.move_bonus
    if args.step_penalty is not None:
        cfg.step_penalty = args.step_penalty
    if args.use_noisy:
        cfg.use_noisy = True
    if args.noisy_sigma is not None:
        cfg.noisy_sigma_init = args.noisy_sigma

    device = get_device(args.device)
    out = RESULTS / args.run
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(
        json.dumps({**cfg.to_dict(), "device": str(device)}, indent=2))
    print(f"[train] run={args.run} device={device} total_steps={cfg.total_steps}")

    env = make_env(seed=cfg.seed, reward_mode=cfg.reward_mode,
                   material_scale=cfg.material_scale, eval_scale=cfg.eval_scale,
                   shaping_gamma=cfg.shaping_gamma,
                   reward_clip=cfg.reward_clip, win_bonus=cfg.win_bonus,
                   step_penalty=cfg.step_penalty, move_bonus=cfg.move_bonus,
                   agent_white=cfg.agent_white,
                   max_episode_steps=cfg.max_episode_steps,
                   frameskip=cfg.frameskip, sticky_prob=cfg.sticky_prob)
    n_actions = env.action_space.n
    agent = DQNAgent(n_actions, cfg, device)
    buffer = ReplayBuffer(cfg.buffer_size, seed=cfg.seed)

    log_f = open(out / "log.csv", "w", newline="")
    log_w = csv.writer(log_f)
    log_w.writerow(["episode", "global_step", "ep_reward", "ep_native",
                    "ep_material", "ep_moves", "ep_len", "epsilon", "mean_loss",
                    "reward_ma100"])
    eval_f = open(out / "eval.csv", "w", newline="")
    eval_w = csv.writer(eval_f)
    eval_w.writerow(["global_step", "eval_reward", "eval_material", "eval_len"])

    ram, info = env.reset(seed=cfg.seed)
    ep_reward = ep_native = 0.0
    ep_len = ep_moves = 0
    episode = 0
    losses = deque(maxlen=1000)
    reward_hist = deque(maxlen=100)
    best_eval = -1e9
    t0 = time.time()

    for step in range(1, cfg.total_steps + 1):
        a = agent.act(ram, step)
        next_ram, r, term, trunc, info = env.step(a)
        done = term or trunc
        buffer.add(ram, a, r, next_ram, done)
        ram = next_ram
        ep_reward += r
        ep_native += info.get("native_reward", 0.0)
        ep_moves += int(info.get("board_changed", False))
        ep_len += 1

        if len(buffer) >= cfg.learning_starts and step % cfg.train_freq == 0:
            losses.append(agent.learn(buffer.sample(cfg.batch_size)))
        if step % cfg.target_update_freq == 0:
            agent.update_target()

        if done:
            reward_hist.append(ep_reward)
            mean_loss = float(np.mean(losses)) if losses else 0.0
            log_w.writerow([episode, step, round(ep_reward, 3),
                            round(ep_native, 3),
                            round(info.get("material_balance", 0.0), 3),
                            ep_moves, ep_len, round(agent.epsilon(step), 4),
                            round(mean_loss, 5),
                            round(float(np.mean(reward_hist)), 3)])
            if episode % 10 == 0:
                sps = step / (time.time() - t0)
                print(f"ep={episode:5d} step={step:7d} "
                      f"R={ep_reward:7.2f} ma100={np.mean(reward_hist):7.2f} "
                      f"mat={info.get('material_balance',0):+.1f} "
                      f"len={ep_len:4d} eps={agent.epsilon(step):.3f} "
                      f"loss={mean_loss:.4f} {sps:.0f}sps")
                log_f.flush()
            episode += 1
            ram, info = env.reset()
            ep_reward = ep_native = 0.0
            ep_len = ep_moves = 0

        if step % cfg.eval_freq == 0:
            er, em, el = evaluate(agent, cfg, cfg.eval_episodes)
            eval_w.writerow([step, round(er, 3), round(em, 3), round(el, 1)])
            eval_f.flush()
            print(f"  [eval] step={step} reward={er:.2f} material={em:+.2f} len={el:.0f}")
            if er > best_eval:
                best_eval = er
                agent.save(out / "best.pt")

        if step % cfg.checkpoint_freq == 0:
            agent.save(out / "last.pt")

    agent.save(out / "last.pt")
    log_f.close()
    eval_f.close()
    env.close()
    print(f"[train] concluído em {time.time()-t0:.0f}s. best_eval={best_eval:.2f}")


if __name__ == "__main__":
    main()
