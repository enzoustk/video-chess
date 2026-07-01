"""Boletim de performance para o agente treinado no ChessMoveEnv.

Mede o agente jogando xadrez de verdade (ação = lance de xadrez) contra o
motor Atari (brancas). Compara contra um baseline aleatório-legal (escolhe
uniformemente entre lances legais).

Uso:
    python -m scripts.benchmark_chess --run dqn_chess --episodes 5 --max-moves 30
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from chess_rl.agent import get_device
from chess_rl.chess_env import N_ACTIONS, VideoChessMoveEnv, move_to_action
from chess_rl.network import DuelingDQN

RESULTS = Path(__file__).resolve().parent.parent / "results"


def load_agent(path, device, noisy=False):
    net = DuelingDQN(N_ACTIONS, noisy=noisy).to(device)
    ckpt = torch.load(path, map_location=device)
    net.load_state_dict(ckpt["online"])
    net.eval()
    return net


def act_greedy(net, ram, mask, device):
    from chess_rl.encoding import encode_batch
    b, a = encode_batch(np.asarray(ram)[None, :])
    with torch.no_grad():
        q = net(torch.from_numpy(b).to(device),
                torch.from_numpy(a).to(device)).cpu().numpy()[0]
    q_masked = np.full_like(q, -np.inf)
    q_masked[mask] = q[mask]
    return int(q_masked.argmax())


def run(policy_fn, episodes, max_moves, seed_base=90_000):
    env = VideoChessMoveEnv(max_moves=max_moves)
    stats = []
    for ep in range(episodes):
        ram, info = env.reset(seed=seed_base + ep)
        mask = info["action_mask"]
        ep_r = 0.0; ep_moves = 0; captures = 0
        prev_mat = info["material"]
        while True:
            a = policy_fn(ram, mask)
            ram, r, term, trunc, info = env.step(a)
            mask = info["action_mask"]
            ep_r += r
            ep_moves += 1
            if info["material"] != prev_mat:
                captures += 1
                prev_mat = info["material"]
            if term or trunc: break
        stats.append(dict(reward=ep_r, moves=ep_moves,
                          final_material=info["material"],
                          final_phi=info["phi"], captures=captures,
                          end_reason=info.get("end_reason"),
                          winner=info.get("winner")))
    env.close()
    return stats


def summarize(name, stats):
    r = [s["reward"] for s in stats]
    m = [s["final_material"] for s in stats]
    p = [s["final_phi"] for s in stats]
    mv = [s["moves"] for s in stats]
    cp = [s["captures"] for s in stats]
    # win/draw/loss (perspectiva agente = pretas)
    wins = sum(1 for s in stats if s.get("winner") == "black")
    draws = sum(1 for s in stats if s.get("winner") == "draw")
    losses = sum(1 for s in stats if s.get("winner") == "white")
    truncated = sum(1 for s in stats if s.get("end_reason") == "truncated")
    n = len(stats)
    winrate = 100 * wins / max(1, n)
    print(f"{name:22s} | W/D/L = {wins}/{draws}/{losses} (winrate={winrate:.0f}%) "
          f"| truncados={truncated}/{n} | material medio={np.mean(m):+5.2f} "
          f"| lances medio={np.mean(mv):5.1f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=None,
                   help="checkpoint do agente (pula se omitido)")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--max-moves", type=int, default=250,
                   help="limite prático de lances; use 0 pra ilimitado")
    p.add_argument("--device", default=None)
    p.add_argument("--lookahead", type=int, default=0,
                   help="se >0, roda também heurística+minimax nessa profundidade")
    args = p.parse_args()

    device = get_device(args.device)
    print(f"Benchmark ChessMove: {args.episodes} eps × {(None if args.max_moves == 0 else args.max_moves)} lances/ep\n")

    rng = np.random.default_rng(0)
    def random_policy(ram, mask):
        legals = np.where(mask)[0]
        return int(rng.choice(legals)) if len(legals) else 0

    print("== BASELINE aleatório-legal ==")
    summarize("aleatorio-legal", run(random_policy, args.episodes, (None if args.max_moves == 0 else args.max_moves)))

    if args.run:
        ckpt_dir = RESULTS / args.run
        cfg = json.loads((ckpt_dir / "config.json").read_text())
        noisy = cfg.get("use_noisy", False)
        net = load_agent(ckpt_dir / "last.pt", device, noisy=noisy)
        def agent_policy(ram, mask):
            return act_greedy(net, ram, mask, device)
        print(f"\n== AGENTE {args.run} (1-ply Q-value) ==")
        summarize(args.run, run(agent_policy, args.episodes, (None if args.max_moves == 0 else args.max_moves)))

    if args.lookahead > 0:
        from chess_rl.lookahead import minimax_move
        def heur_policy(ram, mask):
            return minimax_move(ram, mask, depth=args.lookahead, agent_black=True)
        print(f"\n== HEURÍSTICA + MINIMAX ({args.lookahead}-ply) ==")
        summarize(f"heuristic-d{args.lookahead}",
                  run(heur_policy, args.episodes, (None if args.max_moves == 0 else args.max_moves)))


if __name__ == "__main__":
    main()
