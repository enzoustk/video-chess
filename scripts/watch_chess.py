"""Assistir o agente chess-move jogando ao vivo (imprime cada lance + tabuleiro).

Uso:
    python -m scripts.watch_chess --run dqn_chess_mat_long --max-moves 40
    python -m scripts.watch_chess --run dqn_chess_mat_long --max-moves 40 --random  # baseline
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import chess
import numpy as np
import torch

from chess_rl.agent import get_device
from chess_rl.chess_env import N_ACTIONS, VideoChessMoveEnv, action_to_squares
from chess_rl.network import DuelingDQN
from chess_rl.scripted_moves import board_to_python_chess

RESULTS = Path(__file__).resolve().parent.parent / "results"


def load(path, device, noisy=False):
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
    q[~mask] = -np.inf
    return int(q.argmax())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--max-moves", type=int, default=40)
    p.add_argument("--random", action="store_true",
                   help="usa política aleatória-legal em vez do agente")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = get_device(args.device)
    ckpt_dir = RESULTS / args.run
    cfg = json.loads((ckpt_dir / "config.json").read_text())
    noisy = cfg.get("use_noisy", False)
    net = None if args.random else load(ckpt_dir / "last.pt", device, noisy=noisy)
    print(f"Assistindo {'RANDOM' if args.random else args.run}, {args.max_moves} lances máx.\n")

    env = VideoChessMoveEnv(max_moves=args.max_moves, reward_mode="material")
    ram, info = env.reset(seed=42)
    board = board_to_python_chess(ram, black_to_move=True)
    print("Posição inicial:")
    print(board, "\n")

    rng = np.random.default_rng(0)
    for step in range(args.max_moves):
        mask = info["action_mask"]
        if args.random:
            legals_idx = np.where(mask)[0]
            action = int(rng.choice(legals_idx))
        else:
            action = act_greedy(net, ram, mask, device)
        from_sq, to_sq = action_to_squares(action)
        # peça movida antes da execução
        my_move = chess.Move(from_sq, to_sq)
        try:
            move_san = board.san(my_move)
        except Exception:
            move_san = my_move.uci()
        ram, r, term, trunc, info = env.step(action)
        # reconstrói a posição atual e detecta lance do branco
        new_board = board_to_python_chess(ram, black_to_move=True)
        # imprimir lances
        print(f"── Turno {step+1}  Material {info['material']:+.1f}  Reward: {r:+.2f}  "
              f"exec={info.get('exec_ok', '?')}  branco_respondeu={info.get('engine_responded', '?')}")
        print(f"  Pretas escolheu: {move_san}")
        # tenta identificar o lance do branco por difference
        white_moved = ""
        for sq in range(64):
            b_before = board.piece_at(sq)
            b_after = new_board.piece_at(sq)
            if b_before != b_after and (b_before and b_before.color == chess.WHITE
                                         or b_after and b_after.color == chess.WHITE):
                if b_before and b_before.color == chess.WHITE and not b_after:
                    src = sq
                elif b_after and b_after.color == chess.WHITE and (not b_before or b_before.color != chess.WHITE):
                    dst = sq
        try:
            if 'src' in dir() and 'dst' in dir():
                white_moved = chess.Move(src, dst).uci()
        except Exception:
            pass
        if white_moved:
            print(f"  Brancas: {white_moved}")
        print(new_board)
        print()
        board = new_board
        if term:
            print("Fim de partida (sem lances legais).")
            break
        if trunc:
            print(f"Truncado em {args.max_moves} lances.")
            break
    print(f"\n╔ FIM: material final = {info['material']:+.1f} pawns")
    env.close()


if __name__ == "__main__":
    main()
